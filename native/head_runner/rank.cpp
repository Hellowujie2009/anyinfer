#include <acl/acl.h>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
using Clock = std::chrono::steady_clock;
double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void check(aclError err, const char *call) {
    if (err == ACL_SUCCESS) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(call) + " returned " + std::to_string(err) +
                             (detail ? ": " + std::string(detail) : ""));
}
#define ACL(call) check((call), #call)

uint32_t number(const std::string &value) {
    uint32_t n = 0;
    const auto end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, n);
    require(parsed.ec == std::errc{} && parsed.ptr == end, "invalid integer");
    return n;
}
size_t bytes(uint64_t a, uint64_t b) {
    require(b > 0 && a > 0 && a <= std::numeric_limits<size_t>::max() / b,
            "invalid tensor byte length");
    return static_cast<size_t>(a * b);
}
template<class T>
std::vector<T> read_binary(const char *path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    std::vector<T> data(count);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes(count, sizeof(T))));
    return data;
}

struct IO {
    aclmdlDataset *dataset = nullptr;
    aclDataBuffer *buffer = nullptr;
    void *device = nullptr;
    void init(size_t size) {
        dataset = aclmdlCreateDataset();
        require(dataset != nullptr, "aclmdlCreateDataset failed");
        ACL(aclrtMalloc(&device, size, ACL_MEM_MALLOC_HUGE_FIRST));
        buffer = aclCreateDataBuffer(device, size);
        require(buffer != nullptr, "aclCreateDataBuffer failed");
        ACL(aclmdlAddDatasetBuffer(dataset, buffer));
    }
    void close() {
        if (buffer) aclDestroyDataBuffer(buffer);
        if (dataset) aclmdlDestroyDataset(dataset);
        if (device) aclrtFree(device);
    }
};

struct Session {
    int32_t device = -1;
    bool initialized = false, selected = false, loaded = false;
    uint32_t model = 0;
    aclmdlDesc *desc = nullptr;
    aclrtStream stream = nullptr;
    aclrtEvent begin = nullptr, end = nullptr;
    IO input, output;

    void open(int32_t id, const char *path, uint32_t hc, uint32_t hidden, uint32_t vocab) {
        device = id;
        ACL(aclInit(nullptr));
        initialized = true;
        ACL(aclrtSetDevice(device));
        selected = true;
        aclrtRunMode mode;
        ACL(aclrtGetRunMode(&mode));
        require(mode == ACL_HOST, "host-mode Ascend accelerator required");
        ACL(aclrtCreateStream(&stream));
        ACL(aclmdlLoadFromFile(path, &model));
        loaded = true;
        desc = aclmdlCreateDesc();
        require(desc != nullptr, "aclmdlCreateDesc failed");
        ACL(aclmdlGetDesc(desc, model));
        require(aclmdlGetNumInputs(desc) == 1 && aclmdlGetNumOutputs(desc) == 1, "expected one input and one output");
        aclmdlIODims i{}, o{};
        ACL(aclmdlGetInputDims(desc, 0, &i));
        ACL(aclmdlGetOutputDims(desc, 0, &o));
        require(i.dimCount == 2 && i.dims[0] == hc && i.dims[1] == hidden &&
                o.dimCount == 1 && o.dims[0] == vocab, "OM tensor dimensions disagree with head manifest");
        require(aclmdlGetInputDataType(desc, 0) == ACL_BF16 && aclmdlGetOutputDataType(desc, 0) == ACL_FLOAT,
                "expected BF16 residual and FP32 logits");
        require(aclmdlGetInputFormat(desc, 0) == ACL_FORMAT_ND && aclmdlGetOutputFormat(desc, 0) == ACL_FORMAT_ND,
                "head OM must expose plain ND input/output layouts");
        const size_t in_size = bytes(bytes(hc, hidden), sizeof(uint16_t));
        const size_t out_size = bytes(vocab, sizeof(float));
        require(aclmdlGetInputSizeByIndex(desc, 0) == in_size &&
                aclmdlGetOutputSizeByIndex(desc, 0) == out_size, "OM requires non-dense or padded host tensors");
        input.init(in_size);
        output.init(out_size);
        ACL(aclrtCreateEventExWithFlag(&begin, ACL_EVENT_TIME_LINE));
        ACL(aclrtCreateEventExWithFlag(&end, ACL_EVENT_TIME_LINE));
    }
    void execute() { ACL(aclmdlExecuteAsync(model, input.dataset, output.dataset, stream)); }
    ~Session() {
        if (stream) aclrtSynchronizeStream(stream);
        if (end) aclrtDestroyEvent(end);
        if (begin) aclrtDestroyEvent(begin);
        output.close();
        input.close();
        if (desc) aclmdlDestroyDesc(desc);
        if (loaded) aclmdlUnload(model);
        if (stream) aclrtDestroyStream(stream);
        if (selected) aclrtResetDevice(device);
        if (initialized) aclFinalize();
    }
};

int run(int argc, char **argv) {
    require(argc == 10, "invoke this internal worker via anyinfer run-head");
    const uint32_t device = number(argv[1]), hc = number(argv[2]), hidden = number(argv[3]);
    const uint32_t first = number(argv[4]), vocab = number(argv[5]);
    const uint32_t warmup = number(argv[6]), samples = number(argv[7]);
    require(device <= INT32_MAX && samples > 0 && vocab > 0 && first <= UINT32_MAX - vocab, "invalid rank arguments");
    Session s;
    s.open(static_cast<int32_t>(device), argv[8], hc, hidden, vocab);
    // Each local process uploads the same last-token residual before warm-up.
    const auto residual = read_binary<uint16_t>(argv[9], bytes(hc, hidden));
    for (const uint16_t x : residual)
        require((x & 0x7f80u) != 0x7f80u, "non-finite residual");
    const size_t in_size = bytes(residual.size(), sizeof(uint16_t));
    const auto upload_begin = Clock::now();
    ACL(aclrtMemcpy(s.input.device, in_size, residual.data(), in_size, ACL_MEMCPY_HOST_TO_DEVICE));
    const auto upload_end = Clock::now();
    for (uint32_t i = 0; i < warmup; ++i) s.execute();
    ACL(aclrtSynchronizeStream(s.stream));
    std::cout << "ANYINFER_HEAD_READY\n" << std::flush;
    std::string command;
    require(bool(std::getline(std::cin, command)) && command == "RUN", "missing coordinator start");
    std::vector<float> device_us;
    device_us.reserve(samples);
    for (uint32_t i = 0; i < samples; ++i) {
        ACL(aclrtRecordEvent(s.begin, s.stream));
        s.execute();
        ACL(aclrtRecordEvent(s.end, s.stream));
        ACL(aclrtSynchronizeStream(s.stream));
        float ms = 0;
        ACL(aclrtEventElapsedTime(&ms, s.begin, s.end));
        require(std::isfinite(ms) && ms > 0, "invalid device interval");
        device_us.push_back(ms * 1000);
    }
    // One vector per rank is copied once, after all timed iterations, for full precision checking.
    std::vector<float> logits(vocab);
    const size_t out_size = bytes(vocab, sizeof(float));
    const auto copy_begin = Clock::now();
    ACL(aclrtMemcpy(logits.data(), out_size, s.output.device, out_size, ACL_MEMCPY_DEVICE_TO_HOST));
    const auto copy_end = Clock::now();
    uint32_t winner = 0;
    for (uint32_t i = 0; i < vocab; ++i) {
        require(std::isfinite(logits[i]), "non-finite output logit");
        if (logits[i] > logits[winner]) winner = i;
    }
    const auto id = first + winner;
    const auto reduce_end = Clock::now();
    const char *soc = aclrtGetSocName();
    require(soc && *soc, "aclrtGetSocName returned no name");
    const std::string soc_name(soc);
    require(soc_name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_- .") == std::string::npos,
            "unexpected characters in SoC name");
    std::cout << "ANYINFER_HEAD_RESULT " << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "{\"token_id\":" << id << ",\"logit\":" << logits[winner] << ",\"soc\":\"" << soc_name
              << "\",\"input_h2d_us\":" << elapsed_us(upload_begin, upload_end)
              << ",\"output_d2h_us\":" << elapsed_us(copy_begin, copy_end)
              << ",\"local_checked_argmax_us\":" << elapsed_us(copy_end, reduce_end)
              << ",\"device_us\":[";
    for (size_t i = 0; i < device_us.size(); ++i) std::cout << (i ? "," : "") << device_us[i];
    std::cout << "]}\n" << std::flush;
    // Only send the full shard to the Rust host when a golden comparison was requested.
    if (std::getline(std::cin, command) && command == "LOGITS") {
        std::cout.write(reinterpret_cast<const char *>(logits.data()), static_cast<std::streamsize>(out_size));
        std::cout.flush();
    }
    require(bool(std::cout), "could not write rank output");
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    try { return run(argc, argv); }
    catch (const std::exception &e) {
        std::cout << "ANYINFER_HEAD_ERROR " << e.what() << '\n' << std::flush;
        std::cerr << "anyinfer-head-rank: " << e.what() << '\n';
        return 1;
    }
}
