//! Independent scalar accuracy reference. All I/O is little endian.
use super::Checkpoint;
use super::head::parameter;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;

fn bfloat16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

fn decode(raw: &[u8], bytes_per_value: usize, output: &mut [f32]) -> Result<()> {
    ensure!(
        raw.len()
            == output
                .len()
                .checked_mul(bytes_per_value)
                .context("weight length overflows")?,
        "invalid weight byte count"
    );
    for (cell, bytes) in output.iter_mut().zip(raw.chunks_exact(bytes_per_value)) {
        *cell = if bytes_per_value == 2 {
            f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into()?)) << 16)
        } else {
            f32::from_le_bytes(bytes.try_into()?)
        };
        ensure!(cell.is_finite(), "non-finite residual or weight");
    }
    Ok(())
}

fn small_weight(c: &Checkpoint, names: &[&str], shape: &[u64]) -> Result<Vec<f32>> {
    let (_, t) = parameter(c, names, shape)?;
    let width = if t.dtype == "BF16" { 2 } else { 4 };
    let mut data = vec![0; usize::try_from(t.byte_len)?];
    t.read_range(0, t.byte_len)?.read_exact(&mut data)?;
    let mut weight = vec![0.0; data.len() / width];
    decode(&data, width, &mut weight)?;
    Ok(weight)
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    // A linear FP32 accumulator introduces a length-dependent bias at real V4 widths.
    // Round the reference reduction once, before the FP32 scalar operations.
    let squares: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let mean = (squares / x.len() as f64) as f32;
    (mean + eps).sqrt().recip()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>() as f32
}

#[cfg(test)]
#[test]
fn reference_reductions_retain_small_terms_at_v4_width() {
    let mut x = vec![0.01f32; 16_384];
    x[0] = 100.0;
    let reference_mean = (10_000.0f64 + 16_383.0 * f64::from(0.01f32).powi(2)) / 16_384.0;
    assert_eq!(
        inv_rms(&x, 1e-6),
        ((reference_mean as f32) + 1e-6).sqrt().recip()
    );
    assert_eq!(dot(&[16_777_216., 1., -16_777_216.], &[1., 1., 1.]), 1.);
}

pub fn golden_head(root: &Path, residual: &Path, output: &Path) -> Result<Value> {
    let ckpt = Checkpoint::open(root)?;
    ensure!(
        ckpt.model_type == "deepseek_v4",
        "this output head implements deepseek_v4"
    );
    let h = ckpt.hc_mult;
    let d = ckpt.hidden_size;
    let flat = h.checked_mul(d).context("residual shape overflows")?;
    let bytes = flat.checked_mul(2).context("residual size overflows")?;
    let file = File::open(residual)?;
    ensure!(
        file.metadata()?.len() == bytes,
        "residual must contain exactly {bytes} little-endian BF16 bytes"
    );
    let mut raw = vec![0; usize::try_from(bytes)?];
    (&file).read_exact(&mut raw)?;
    let mut x = vec![0.0; usize::try_from(flat)?];
    decode(&raw, 2, &mut x)?;
    let gate_weight = small_weight(&ckpt, &["hc_head_fn", "model.hc_head_fn"], &[h, flat])?;
    let scale = small_weight(&ckpt, &["hc_head_scale", "model.hc_head_scale"], &[1])?[0];
    let bias = small_weight(&ckpt, &["hc_head_base", "model.hc_head_base"], &[h])?;
    let norm = small_weight(&ckpt, &["norm.weight", "model.norm.weight"], &[d])?;
    let mut combined = vec![0.0; usize::try_from(d)?];
    let inv = inv_rms(&x, ckpt.norm_eps);
    for ((w, b), stream) in gate_weight
        .chunks_exact(x.len())
        .zip(bias)
        .zip(x.chunks_exact(norm.len()))
    {
        let score = dot(w, &x) * inv * scale + b;
        let gate = (1.0 + (-score).exp()).recip() + ckpt.hc_eps;
        for (out, &v) in combined.iter_mut().zip(stream) {
            *out += v * gate;
        }
    }
    for v in &mut combined {
        *v = bfloat16(*v);
    }
    let inv = inv_rms(&combined, ckpt.norm_eps);
    for (v, k) in combined.iter_mut().zip(norm) {
        *v = bfloat16((*v * inv) * k);
    }
    let vocab = ckpt.vocab_size;
    let (_, lm) = parameter(
        &ckpt,
        &[
            "head.weight",
            "model.head.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ],
        &[vocab, d],
    )?;
    let width = if lm.dtype == "BF16" { 2 } else { 4 };
    let mut lm_reader = BufReader::with_capacity(1 << 20, lm.read_range(0, lm.byte_len)?);
    let mut row = vec![0.0; combined.len()];
    let mut row_bytes = vec![0; usize::try_from(lm.byte_len / vocab)?];
    let mut logits = Vec::with_capacity(usize::try_from(
        vocab.checked_mul(4).context("logits size overflows")?,
    )?);
    let mut best = (0, f32::NEG_INFINITY);
    for token in 0..vocab {
        lm_reader.read_exact(&mut row_bytes)?;
        decode(&row_bytes, width, &mut row)?;
        let logit = dot(&combined, &row);
        ensure!(
            logit.is_finite(),
            "non-finite reference logit for token {token}"
        );
        if logit > best.1 {
            best = (token, logit);
        }
        logits.extend_from_slice(&logit.to_le_bytes());
    }
    let mut writer =
        File::create_new(output).with_context(|| format!("create {}", output.display()))?;
    if let Err(error) = writer.write_all(&logits).and_then(|_| writer.sync_all()) {
        drop(writer);
        let _ = fs::remove_file(output);
        return Err(error.into());
    }
    Ok(
        json!({"model_type":ckpt.model_type,"vocab_size":vocab,"dtype":"F32","bytes":logits.len(),"token_id":best.0,"logit":best.1}),
    )
}
