#include "acl/acl.h"
#include "hc_post.h"
#include "reference.h"
#include "shape.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(aclError code, const char *name) {
    if (code == ACL_SUCCESS) return;
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(name) + " returned " + std::to_string(code) +
                             (detail ? std::string(" ") + detail : ""));
}
#define CHECK_ACL(expr) check((expr), #expr)

struct Options {
    uint32_t device = 0, tokens = 0, hidden = 0, warmup = 5, samples = 7, iterations = 8, d2d_mib = 64, tile = 0;
    std::string output;
};
void parse(Options &o, int argc, char **argv) {
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[i + 1];
        if (key == "--output") { o.output = value; continue; }
        uint32_t n = 0;
        const auto end = value.data() + value.size();
        const auto p = std::from_chars(value.data(), end, n);
        if (p.ec != std::errc{} || p.ptr != end) throw std::invalid_argument("invalid integer for " + key);
        if (key == "--device") o.device = n;
        else if (key == "--tokens") o.tokens = n;
        else if (key == "--hidden") o.hidden = n;
        else if (key == "--warmup") o.warmup = n;
        else if (key == "--samples") o.samples = n;
        else if (key == "--iterations") o.iterations = n;
        else if (key == "--d2d-mib") o.d2d_mib = n;
        else if (key == "--tile") {
            if (!hc_valid_tile(n)) throw std::invalid_argument("--tile must be 128, 256, 512 or 1024");
            o.tile = n;
        }
        else throw std::invalid_argument("unknown option " + key);
    }
    if (o.output.empty() || !hc_valid_shape(o.tokens, o.hidden) || !o.samples || !o.iterations ||
        !o.d2d_mib || o.device > INT32_MAX)
        throw std::invalid_argument("expected --tokens T --hidden H --output FILE, H divisible by 16, positive sizes");
}

std::string json_string(const std::string &s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
void save(const std::string &path, const std::string &result) {
    const std::string tmp = path + ".part";
    std::ofstream file(tmp);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << result << '\n';
    file.close();
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot publish " + path);
}
std::string soc_key(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) * 0.5;
}
std::string numbers(const std::vector<double> &samples) {
    std::ostringstream out;
    out << std::setprecision(12) << '[';
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) out << ',';
        out << samples[i];
    }
    out << ']';
    return out.str();
}

class Device {
public:
    aclrtStream stream = nullptr;
    void open(int32_t id) {
        device_ = id;
        CHECK_ACL(aclInit(nullptr));
        init_ = true;
        CHECK_ACL(aclrtSetDevice(id));
        selected_ = true;
        CHECK_ACL(aclrtCreateStream(&stream));
    }
    void *allocate(size_t bytes) {
        buffers_.push_back(nullptr);
        CHECK_ACL(aclrtMalloc(&buffers_.back(), bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        return buffers_.back();
    }
    template<class T> void *upload(const std::vector<T> &data) {
        const size_t bytes = data.size() * sizeof(T);
        void *dst = allocate(bytes);
        CHECK_ACL(aclrtMemcpy(dst, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        return dst;
    }
    ~Device() {
        if (stream) aclrtSynchronizeStream(stream);
        for (auto p : buffers_) if (p) aclrtFree(p);
        if (stream) aclrtDestroyStream(stream);
        if (selected_) aclrtResetDevice(device_);
        if (init_) aclFinalize();
    }
private:
    bool init_ = false, selected_ = false;
    int32_t device_ = 0;
    std::vector<void *> buffers_;
};

struct Event {
    aclrtEvent handle = nullptr;
    Event() { CHECK_ACL(aclrtCreateEventExWithFlag(&handle, ACL_EVENT_TIME_LINE)); }
    ~Event() { aclrtDestroyEvent(handle); }
};
template<class F> double time_us(aclrtStream stream, uint32_t iterations, F launch) {
    Event start, end;
    CHECK_ACL(aclrtRecordEvent(start.handle, stream));
    for (uint32_t i = 0; i < iterations; ++i) launch();
    CHECK_ACL(aclrtRecordEvent(end.handle, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    float ms = 0;
    CHECK_ACL(aclrtEventElapsedTime(&ms, start.handle, end.handle));
    if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("invalid device event interval");
    return double(ms) * 1000 / iterations;
}

bool run(const Options &o) {
    Device device;
    device.open(static_cast<int32_t>(o.device));
    const char *soc_name = aclrtGetSocName();
    if (!soc_name) throw std::runtime_error("aclrtGetSocName returned null");
    const std::string soc = soc_name;
    if (soc_key(soc) != soc_key(ANYINFER_SOC))
        throw std::runtime_error("runtime SoC " + soc + " differs from compiled SoC " ANYINFER_SOC);
    int64_t cores = 0;
    CHECK_ACL(aclGetDeviceCapability(o.device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores));
    if (cores <= 0 || cores > UINT32_MAX) throw std::runtime_error("invalid vector core count");
    const uint32_t tile = o.tile ? o.tile : hc_tile(o.tokens);
    const auto blocks = static_cast<uint32_t>(std::min<uint64_t>(cores, hc_work(o.tokens, o.hidden, tile)));
    reference::Inputs input(o.tokens, o.hidden);
    void *x = device.upload(input.x);
    void *r = device.upload(input.residual);
    void *post = device.upload(input.post);
    void *comb = device.upload(input.comb);
    const size_t elements = input.residual.size();
    const size_t bytes = elements * sizeof(uint16_t);
    void *y = device.allocate(bytes);
    CHECK_ACL(aclrtMemset(y, bytes, 0x7f, bytes));
    auto launch = [&] {
        CHECK_ACL(anyinfer_hc_post_launch_tiled(device.stream, blocks, o.tokens, o.hidden, tile, x, r, post, comb, y));
    };
    for (uint32_t i = 0; i < o.warmup; ++i) launch();
    CHECK_ACL(aclrtSynchronizeStream(device.stream));
    std::vector<double> samples, wall;
    for (uint32_t i = 0; i < o.samples; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        samples.push_back(time_us(device.stream, o.iterations, launch));
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        wall.push_back(std::chrono::duration<double, std::micro>(elapsed).count() / o.iterations);
    }
    std::vector<uint16_t> output(elements);
    CHECK_ACL(aclrtMemcpy(output.data(), bytes, y, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    uint64_t mismatches = 0;
    double max_error = 0;
    for (size_t t = 0; t < o.tokens; ++t) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t d = 0; d < o.hidden; ++d) {
                const float expected = reference::fp32(input.expected(t, j, d, o.hidden));
                const float actual = reference::fp32(output[(t * 4 + j) * o.hidden + d]);
                if (expected != actual) ++mismatches;
                if (!std::isfinite(actual)) max_error = std::numeric_limits<double>::infinity();
                else max_error = std::max(max_error, double(std::abs(actual - expected)));
            }
        }
    }
    // Read+write bytes for a same-device asynchronous copy on the same stream.
    // Reusing both buffers makes this an empirical comparator sensitive to device caching.
    const size_t copy_bytes = size_t(o.d2d_mib) << 20;
    void *copy_src = device.allocate(copy_bytes);
    void *copy_dst = device.allocate(copy_bytes);
    CHECK_ACL(aclrtMemset(copy_src, copy_bytes, 0x5a, copy_bytes));
    auto copy = [&] {
        CHECK_ACL(aclrtMemcpyAsync(copy_dst, copy_bytes, copy_src, copy_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, device.stream));
    };
    copy();
    CHECK_ACL(aclrtSynchronizeStream(device.stream));
    std::vector<double> copy_samples;
    for (uint32_t i = 0; i < 3; ++i) copy_samples.push_back(time_us(device.stream, 4, copy));

    const double us = median(samples);
    const uint64_t logical = uint64_t(o.tokens) * (18ull * o.hidden + 80);
    const double gb_s = double(logical) / (us * 1000);
    const double d2d = 2.0 * copy_bytes / (median(copy_samples) * 1000);
    std::ostringstream out;
    out << std::setprecision(12) << "{\"schema_version\":1,\"operator\":\"hc_post\",\"status\":"
        << json_string(mismatches ? "failed" : "passed") << ",\"device\":" << o.device
        << ",\"soc\":" << json_string(soc) << ",\"compiled_soc\":" << json_string(ANYINFER_SOC)
        << ",\"tokens\":" << o.tokens << ",\"hidden\":" << o.hidden << ",\"vector_cores\":" << cores
        << ",\"blocks\":" << blocks << ",\"tile_elements\":" << tile << ",\"ub_bytes_per_core\":" << hc_ub_bytes(tile)
        << ",\"warmup\":" << o.warmup << ",\"iterations\":" << o.iterations
        << ",\"samples_us\":" << numbers(samples) << ",\"median_us\":" << us
        << ",\"wall_samples_us\":" << numbers(wall) << ",\"logical_bytes\":" << logical
        << ",\"effective_gb_s\":" << gb_s << ",\"d2d_copy_bytes\":" << copy_bytes
        << ",\"d2d_working_set_bytes\":" << 2 * copy_bytes << ",\"d2d_samples_us\":" << numbers(copy_samples)
        << ",\"d2d_gb_s\":" << d2d << ",\"effective_to_d2d_ratio\":" << gb_s / d2d
        << ",\"compared_elements\":" << elements << ",\"mismatches\":" << mismatches << ",\"max_abs_error\":";
    if (std::isfinite(max_error)) out << max_error;
    else out << "null";
    out << '}';
    save(o.output, out.str());
    return mismatches == 0;
}
}

int main(int argc, char **argv) {
    Options options;
    try {
        parse(options, argc, argv);
        return run(options) ? 0 : 1;
    } catch (const std::exception &err) {
        std::cerr << "hc_post benchmark: " << err.what() << '\n';
        if (!options.output.empty()) {
            try {
                save(options.output, "{\"schema_version\":1,\"operator\":\"hc_post\",\"status\":\"failed\",\"error\":"
                    + json_string(err.what()) + ",\"compiled_soc\":" + json_string(ANYINFER_SOC)
                    + ",\"device\":" + std::to_string(options.device) + ",\"tokens\":" + std::to_string(options.tokens)
                    + ",\"hidden\":" + std::to_string(options.hidden) + "}");
            } catch (const std::exception &io) { std::cerr << io.what() << '\n'; }
        }
        return 1;
    }
}
