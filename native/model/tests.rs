use super::*;
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "anyinfer-model-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let cfg = json!({"model_type":"deepseek_v4","hc_mult":2,"hidden_size":2,"vocab_size":4,"rms_norm_eps":1e-6,"hc_eps":1e-6});
        fs::write(path.join("config.json"), cfg.to_string()).unwrap();
        Self(path)
    }
    fn shard(&self, name: &str, entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) {
        let mut header = BTreeMap::new();
        let mut payload = Vec::new();
        for (key, dtype, shape, data) in entries {
            let start = payload.len();
            payload.extend_from_slice(data);
            header.insert(
                *key,
                json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
            );
        }
        let mut hdr = serde_json::to_vec(&header).unwrap();
        hdr.resize(hdr.len().div_ceil(8) * 8, b' ');
        let mut out = File::create(self.0.join(name)).unwrap();
        out.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        out.write_all(&hdr).unwrap();
        out.write_all(&payload).unwrap();
    }
    fn index(&self, weight_map: Value) {
        fs::write(
            self.0.join("model.safetensors.index.json"),
            json!({"weight_map":weight_map}).to_string(),
        )
        .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn native_reader_bounds_and_index_membership() {
    let f = Fixture::new();
    f.shard(
        "a.safetensors",
        &[("quant", "I8", vec![3], vec![7, 13, 23])],
    );
    f.shard(
        "b.safetensors",
        &[("norm.weight", "BF16", vec![2], vec![128, 63, 0, 64])],
    );
    f.index(json!({"quant":"a.safetensors","norm.weight":"b.safetensors"}));
    let c = Checkpoint::open(&f.0).unwrap();
    let mut data = Vec::new();
    c.reader("norm.weight")
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    assert_eq!(data, [128, 63, 0, 64]);
    assert!(c.tensor("quant").unwrap().read_range(2, 2).is_err());
    let info = inspect(&f.0).unwrap();
    assert_eq!(info["shards"], 2);
    assert_eq!(info["dtypes"]["BF16"]["bytes"], 4);
    assert_eq!(info["hc_post_cases"][0]["residual"], json!([1, 2, 2]));
    f.index(json!({"quant":"b.safetensors","norm.weight":"a.safetensors"}));
    assert!(Checkpoint::open(&f.0).is_err());
    f.index(json!({"quant":"../a.safetensors"}));
    assert!(Checkpoint::open(&f.0).is_err());
}

#[test]
fn reject_truncated_or_inconsistent_tensors() {
    let f = Fixture::new();
    f.shard("model.safetensors", &[("bad", "F32", vec![3], vec![0; 8])]);
    assert!(Checkpoint::open(&f.0).is_err());
    f.shard("model.safetensors", &[("good", "F32", vec![2], vec![0; 8])]);
    let file = fs::OpenOptions::new()
        .write(true)
        .open(f.0.join("model.safetensors"))
        .unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    assert!(Checkpoint::open(&f.0).is_err());
}

#[test]
fn head_shards_weights_without_reading_other_vocab_rows() {
    let f = Fixture::new();
    let weights = (0u8..16).collect::<Vec<_>>();
    f.shard(
        "model.safetensors",
        &[
            ("hc_head_fn", "F32", vec![2, 4], vec![0; 32]),
            ("hc_head_scale", "F32", vec![1], vec![0; 4]),
            ("hc_head_base", "F32", vec![2], vec![0; 8]),
            ("model.norm.weight", "BF16", vec![2], vec![0; 4]),
            ("lm_head.weight", "BF16", vec![4, 2], weights.clone()),
        ],
    );
    let out = f.0.join("rank-1");
    let report = prepare_head(&f.0, &out, 1, 2).unwrap();
    assert_eq!(report["vocab_start"], 2);
    assert_eq!(report["vocab_end"], 4);
    assert_eq!(
        fs::read(out.join("vocab_weight.bin")).unwrap(),
        weights[8..]
    );
    assert!(fs::metadata(out.join("head.onnx")).unwrap().len() > 0);
    assert!(prepare_head(&f.0, &out, 1, 2).is_err());
    let invalid = f.0.join("bad");
    assert!(prepare_head(&f.0, &invalid, 0, 3).is_err());
    assert!(!invalid.exists());
}

#[test]
fn golden_logits_include_both_bfloat16_roundings_and_vocab_order() {
    let f = Fixture::new();
    let bf = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    f.shard(
        "model.safetensors",
        &[
            ("hc_head_fn", "F32", vec![2, 4], vec![0; 32]),
            ("hc_head_scale", "F32", vec![1], 1f32.to_le_bytes().to_vec()),
            ("hc_head_base", "F32", vec![2], vec![0; 8]),
            ("norm.weight", "BF16", vec![2], bf(&[2., 0.5])),
            (
                "head.weight",
                "BF16",
                vec![4, 2],
                bf(&[1., 0., 0., 1., 1., 1., -1., 1.]),
            ),
        ],
    );
    let input = f.0.join("residual.bin");
    let output = f.0.join("logits.bin");
    fs::write(&input, bf(&[2., 0., -1., 2.])).unwrap();
    let report = golden_head(&f.0, &input, &output).unwrap();
    assert_eq!(report["token_id"], 2);
    let expected = [1.265625f32, 0.6328125, 1.8984375, -0.6328125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(fs::read(output).unwrap(), expected);
}
