use anyhow::{Result, ensure};
use anyinfer::native::scheduler::{
    Batch, Config, Event, Executor, FinishReason, Scheduler, TokenStream,
};
use serde_json::{Value, json};
use std::time::Instant;
use tokio::sync::mpsc;

// The cost measured here is host bookkeeping for a device submission. The
// benchmark never loads weights, predicts model outputs or waits on hardware.
struct EmptyDispatch;

impl Executor for EmptyDispatch {
    fn step(&mut self, batch: &Batch, next: &mut [u32]) -> Result<()> {
        std::hint::black_box(batch);
        next.fill(1);
        Ok(())
    }
}

fn distribution(ns: &mut [u128]) -> Value {
    ns.sort_unstable();
    let n = ns.len();
    let rank = |per_mille: usize| ns[(n * per_mille).div_ceil(1_000) - 1];
    json!({"count":n, "p50_ns":rank(500), "p95_ns":rank(950), "p99_ns":rank(990),
        "p999_ns":(n >= 1_000).then(|| rank(999)), "max_ns":ns[n-1],
        "sum_ns":ns.iter().sum::<u128>()})
}

fn measure(slots: usize, chunk: usize, prompts: &[usize], generations: &[u32]) -> Result<Value> {
    const JOBS: usize = 256;
    let mut scheduler = Scheduler::new(Config {
        slots,
        prefill_batch: slots.min(16),
        decode_batch: slots,
        prefill_chunk: chunk,
        context_len: 8192,
        vocab_size: 129280,
        eos_tokens: vec![1],
        pad_token: 0,
        max_pending: JOBS,
    })?;
    let mut receivers = Vec::with_capacity(JOBS);
    let mut submissions = Vec::with_capacity(JOBS);
    let mut steps = Vec::new();
    while receivers.len() < JOBS || scheduler.has_work() {
        // New arrivals join while earlier requests still own their cache slots.
        for _ in 0..3.min(JOBS - receivers.len()) {
            let i = receivers.len();
            let prompt = vec![31; prompts[i % prompts.len()]];
            let start = Instant::now();
            let (_, rx) =
                scheduler.submit_stream(prompt, generations[i % generations.len()], true)?;
            submissions.push(start.elapsed().as_nanos());
            receivers.push(rx);
        }
        let start = Instant::now();
        scheduler.tick(&mut EmptyDispatch)?;
        steps.push(start.elapsed().as_nanos());
    }
    let counters = scheduler.stats();
    let mut delivered = 0u64;
    for mut rx in receivers {
        while rx.try_recv().is_ok() {
            delivered += 1;
        }
    }
    let expected: u64 = (0..JOBS)
        .map(|i| u64::from(generations[i % generations.len()]))
        .sum();
    ensure!(
        counters.finished == JOBS as u64 && delivered == expected,
        "scheduler replay lost requests or events"
    );
    Ok(
        json!({"slots":slots,"chunk":chunk,"prompt_lengths":prompts,"generation_lengths":generations,
        "transport":"preallocated","submit":distribution(&mut submissions),"tick":distribution(&mut steps),"counters":counters}),
    )
}

#[derive(Clone, Copy)]
enum Transport {
    Tokio,
    Preallocated,
}

impl Transport {
    fn name(self) -> &'static str {
        match self {
            Self::Tokio => "tokio_unbounded",
            Self::Preallocated => "preallocated",
        }
    }
}

enum Consumer {
    Tokio(mpsc::UnboundedReceiver<Event>),
    Preallocated(TokenStream),
}

impl Consumer {
    fn try_recv(&mut self) -> Result<Event, mpsc::error::TryRecvError> {
        match self {
            Self::Tokio(rx) => rx.try_recv(),
            Self::Preallocated(rx) => rx.try_recv(),
        }
    }
}

// EmptyDispatch emits token 1 in every active row. Read the exact number owed
// to *each* request, including the final length marker, before accepting times.
fn drain(streams: &mut [Consumer], tokens: usize, finished: bool) -> Result<()> {
    for stream in streams {
        for token in 1..=tokens {
            let finish = (finished && token == tokens).then_some(FinishReason::Length);
            ensure!(
                stream.try_recv()? == Event::Token { token: 1, finish },
                "incorrect token or finish marker in host transport replay"
            );
        }
        if finished {
            ensure!(
                matches!(
                    stream.try_recv(),
                    Err(mpsc::error::TryRecvError::Disconnected)
                ),
                "token stream still open or extra event after request completion"
            );
        }
    }
    Ok(())
}

struct Samples {
    submit: Vec<u128>,
    decode: Vec<u128>,
    drain: Vec<u128>,
}

impl Samples {
    fn with_capacity(submissions: usize, ticks: usize, drains: usize) -> Self {
        Self {
            submit: Vec::with_capacity(submissions),
            decode: Vec::with_capacity(ticks),
            drain: Vec::with_capacity(drains),
        }
    }

    fn summarize(&mut self) -> Value {
        json!({"submit":distribution(&mut self.submit), "decode_tick":distribution(&mut self.decode),
            "consumer_drain":distribution(&mut self.drain)})
    }

    fn append(&mut self, run: &mut Self) {
        self.submit.append(&mut run.submit);
        self.decode.append(&mut run.decode);
        self.drain.append(&mut run.drain);
    }
}

fn decode_run(
    slots: usize,
    steady_ticks: usize,
    drain_every: usize,
    mode: Transport,
) -> Result<Samples> {
    let generated = u32::try_from(steady_ticks + 2)?;
    let mut scheduler = Scheduler::new(Config {
        slots,
        prefill_batch: slots,
        decode_batch: slots,
        prefill_chunk: 1,
        context_len: 1 + generated as usize,
        vocab_size: 129280,
        eos_tokens: vec![1],
        pad_token: 0,
        max_pending: slots,
    })?;
    let mut streams = Vec::with_capacity(slots);
    let mut samples =
        Samples::with_capacity(slots, steady_ticks, steady_ticks.div_ceil(drain_every));
    for _ in 0..slots {
        let prompt = vec![31];
        let start = Instant::now();
        let rx = match mode {
            Transport::Tokio => {
                let (tx, rx) = mpsc::unbounded_channel();
                scheduler.submit(prompt, generated, true, tx)?;
                Consumer::Tokio(rx)
            }
            Transport::Preallocated => {
                Consumer::Preallocated(scheduler.submit_stream(prompt, generated, true)?.1)
            }
        };
        samples.submit.push(start.elapsed().as_nanos());
        streams.push(rx);
    }
    // Exclude prefill, prompt reclamation and final request/slot release from
    // steady-state tick samples. Consumer work runs on the same thread between ticks.
    scheduler.tick(&mut EmptyDispatch)?;
    drain(&mut streams, 1, false)?;
    let mut last_drain = 0;
    for tick in 1..=steady_ticks {
        let start = Instant::now();
        scheduler.tick(&mut EmptyDispatch)?;
        samples.decode.push(start.elapsed().as_nanos());
        if tick.is_multiple_of(drain_every) || tick == steady_ticks {
            let start = Instant::now();
            drain(&mut streams, tick - last_drain, false)?;
            samples.drain.push(start.elapsed().as_nanos());
            last_drain = tick;
        }
    }
    scheduler.tick(&mut EmptyDispatch)?;
    drain(&mut streams, 1, true)?;
    let counters = scheduler.stats();
    ensure!(
        !scheduler.has_work()
            && counters.prefill_steps == 1
            && counters.decode_steps == 1 + steady_ticks as u64
            && counters.generated_tokens == slots as u64 * u64::from(generated)
            && counters.finished == slots as u64,
        "incomplete host transport replay"
    );
    Ok(samples)
}

fn paired_decode(drain_every: usize) -> Result<Value> {
    const SLOTS: usize = 8;
    const STEADY: usize = 2048;
    const PAIRS: usize = 8;
    let modes = [Transport::Tokio, Transport::Preallocated];
    let mut totals = modes.map(|_| {
        Samples::with_capacity(
            SLOTS * PAIRS,
            STEADY * PAIRS,
            STEADY.div_ceil(drain_every) * PAIRS,
        )
    });
    for mode in modes {
        decode_run(SLOTS, STEADY, drain_every, mode)?;
    }
    let mut runs = Vec::with_capacity(PAIRS * modes.len());
    for pair in 0..PAIRS {
        for order in 0..2 {
            let index = (pair + order) % 2;
            let mut sample = decode_run(SLOTS, STEADY, drain_every, modes[index])?;
            runs.push(
                json!({"pair":pair+1, "order":order+1, "transport":modes[index].name(),
                "timing":sample.summarize()}),
            );
            totals[index].append(&mut sample);
        }
    }
    let mut pooled = serde_json::Map::new();
    for (mode, mut total) in modes.into_iter().zip(totals) {
        pooled.insert(mode.name().to_string(), total.summarize());
    }
    Ok(
        json!({"slots":SLOTS,"steady_decode_ticks_per_run":STEADY,"decode_batch_rows":SLOTS,
        "generated_tokens_per_request":STEADY+2,"drain_every_ticks":drain_every,
        "warmup_runs_per_transport":1,"pairs":PAIRS,"pooled":pooled,"runs":runs}),
    )
}

pub fn report() -> Result<Value> {
    let mut measurements = Vec::new();
    for (slots, chunk, prompts, generated) in [
        (1, 32, &[1, 17, 33][..], &[1, 4][..]),
        (32, 128, &[32, 127, 128, 129, 1025], &[1, 16, 64]),
        (64, 256, &[2047, 2048, 2049, 4096], &[4, 32]),
    ] {
        measurements.push(measure(slots, chunk, prompts, generated)?);
    }
    Ok(
        json!({"scope":"host_scheduler_microbenchmark", "executor":"empty_dispatch", "measurements":measurements,
        "paired_decode":{
            "submit_scope":"stream creation and scheduler admission; prompt Vec already allocated",
            "decode_tick_scope":"synchronous host scheduler tick including empty dispatch and publishing eight tokens",
            "consumer_drain_scope":"same-thread nonblocking reads with token and finish checks, outside tick",
            "quantiles":"pooled nearest rank; p999_ns null for fewer than 1000 samples",
            "cases":[paired_decode(1)?, paired_decode(64)?]
        }}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p999_uses_nearest_rank_and_requires_enough_samples() {
        assert!(distribution(&mut [1; 999])["p999_ns"].is_null());
        let mut ns: Vec<_> = (1..=1000).rev().collect();
        let times = distribution(&mut ns);
        assert_eq!(times["p999_ns"], 999);
        assert_eq!(times["max_ns"], 1000);
    }

    #[test]
    fn paired_replay_checks_all_events_with_a_partial_final_drain() {
        for mode in [Transport::Tokio, Transport::Preallocated] {
            let samples = decode_run(3, 17, 5, mode).unwrap();
            assert_eq!(
                (
                    samples.submit.len(),
                    samples.decode.len(),
                    samples.drain.len()
                ),
                (3, 17, 4)
            );
        }
    }
}
