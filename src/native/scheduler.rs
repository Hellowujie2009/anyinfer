//! Single-owner scheduler for chunked prefill and one-token decode on a single host.
//! Each request owns one cache slot until its last device step completes.

pub use super::stream::TokenStream;
use super::stream::{self, Writer};
use anyhow::{Result, anyhow, ensure};
use serde::Serialize;
use std::collections::VecDeque;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone)]
pub struct Config {
    pub slots: usize,
    pub prefill_batch: usize,
    pub decode_batch: usize,
    pub prefill_chunk: usize,
    pub context_len: usize,
    pub vocab_size: u32,
    pub eos_tokens: Vec<u32>,
    pub pad_token: u32,
    pub max_pending: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Debug)]
pub struct Batch {
    pub phase: Phase,
    pub width: usize,
    pub slots: Vec<u32>,
    /// Number of tokens already in this slot's cache before this device step.
    pub positions: Vec<u32>,
    pub lengths: Vec<u32>,
    /// Contiguous row-major [slots.len(), width]. Only lengths[row] tokens are valid.
    pub tokens: Vec<u32>,
}

pub trait Executor {
    /// Synchronously finish this step on every participating device. Position zero
    /// starts a fresh sequence, including all model-specific recurrent/cache state.
    /// Write one greedy next-token ID at each active row's final valid position.
    fn step(&mut self, batch: &Batch, next_tokens: &mut [u32]) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Event {
    Token {
        token: u32,
        finish: Option<FinishReason>,
    },
    Failed(String),
}

enum Output {
    Reserved(Writer),
    Channel(UnboundedSender<Event>),
}

impl Output {
    fn is_closed(&self) -> bool {
        match self {
            Self::Reserved(writer) => writer.is_closed(),
            Self::Channel(sender) => sender.is_closed(),
        }
    }

    fn send(&mut self, event: Event) -> Result<(), ()> {
        match self {
            Self::Reserved(writer) => writer.send(event),
            Self::Channel(sender) => sender.send(event).map_err(|_| ()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Stats {
    pub waiting: usize,
    pub active: usize,
    pub admitted: u64,
    pub finished: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub prefill_rows: u64,
    pub prefill_input_tokens: u64,
    /// Padding inside active rows, excluding extra inactive rows in a static device graph.
    pub prefill_row_padding_tokens: u64,
    pub decode_input_tokens: u64,
    pub generated_tokens: u64,
    pub total_queue_wait_us: u128,
    pub first_tokens: u64,
    /// Submission to first valid result, observed on the scheduler host.
    pub total_first_token_us: u128,
    /// Host-side elapsed time waiting for the synchronous executor.
    pub prefill_step_us: u128,
    pub decode_step_us: u128,
}

struct Request {
    id: u64,
    submitted: Instant,
    prompt: Vec<u32>,
    max_new_tokens: u32,
    ignore_eos: bool,
    stream: Output,
    cached: usize,
    generated: u32,
    next_input: u32,
}

pub struct Scheduler {
    config: Config,
    waiting: VecDeque<Request>,
    slots: Vec<Option<Request>>,
    free: Vec<usize>,
    prefill: VecDeque<usize>,
    decode: VecDeque<usize>,
    last_phase: Phase,
    batch: Batch,
    output: Vec<u32>,
    next_id: u64,
    failure: Option<String>,
    counters: Stats,
}

impl Scheduler {
    pub fn new(config: Config) -> Result<Self> {
        ensure!(
            config.slots > 0 && config.slots <= u32::MAX as usize,
            "invalid KV slot count"
        );
        ensure!(
            config.prefill_batch > 0 && config.decode_batch > 0,
            "empty graph batch"
        );
        ensure!(
            config.prefill_chunk > 0
                && config.prefill_chunk <= config.context_len
                && config.context_len <= u32::MAX as usize,
            "invalid context or prefill chunk length"
        );
        ensure!(config.max_pending > 0, "max_pending must be positive");
        ensure!(
            config.pad_token < config.vocab_size
                && config.eos_tokens.iter().all(|&id| id < config.vocab_size),
            "special token outside vocabulary"
        );
        let prefill_rows = config.prefill_batch.min(config.slots);
        let decode_rows = config.decode_batch.min(config.slots);
        let max_rows = prefill_rows.max(decode_rows);
        let tokens = prefill_rows
            .checked_mul(config.prefill_chunk)
            .ok_or_else(|| anyhow!("prefill buffer size overflow"))?
            .max(decode_rows);
        Ok(Self {
            slots: (0..config.slots).map(|_| None).collect(),
            free: (0..config.slots).rev().collect(),
            waiting: VecDeque::new(),
            prefill: VecDeque::with_capacity(config.slots),
            decode: VecDeque::with_capacity(config.slots),
            // The first step admits waiting prompts.
            last_phase: Phase::Decode,
            batch: Batch {
                phase: Phase::Prefill,
                width: 0,
                slots: Vec::with_capacity(max_rows),
                positions: Vec::with_capacity(max_rows),
                lengths: Vec::with_capacity(max_rows),
                tokens: Vec::with_capacity(tokens),
            },
            output: vec![0; max_rows],
            next_id: 0,
            failure: None,
            counters: Stats::default(),
            config,
        })
    }

    pub fn submit(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: u32,
        ignore_eos: bool,
        stream: UnboundedSender<Event>,
    ) -> Result<u64> {
        ensure!(!stream.is_closed(), "request receiver already closed");
        self.validate_submission(&prompt, max_new_tokens)?;
        self.enqueue(prompt, max_new_tokens, ignore_eos, Output::Channel(stream))
    }

    /// Reserves the request's token payload before enqueuing. Publishing tokens
    /// uses the same fixed storage throughout prefill and decode. Both admission
    /// and the token reservation must succeed before the request becomes visible.
    pub fn submit_stream(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: u32,
        ignore_eos: bool,
    ) -> Result<(u64, TokenStream)> {
        self.validate_submission(&prompt, max_new_tokens)?;
        let (writer, receiver) = stream::channel(max_new_tokens)?;
        let id = self.enqueue(prompt, max_new_tokens, ignore_eos, Output::Reserved(writer))?;
        Ok((id, receiver))
    }

    fn validate_submission(&mut self, prompt: &[u32], max_new_tokens: u32) -> Result<()> {
        ensure!(
            self.failure.is_none(),
            "device executor must be restarted after failure"
        );
        ensure!(
            !prompt.is_empty() && max_new_tokens > 0,
            "empty prompt or generation budget"
        );
        ensure!(
            prompt
                .len()
                .checked_add(max_new_tokens as usize)
                .is_some_and(|n| n <= self.config.context_len),
            "prompt and generation exceed context length"
        );
        ensure!(
            prompt.iter().all(|&id| id < self.config.vocab_size),
            "prompt token outside vocabulary"
        );
        // Admit immediately when a slot is free so max_pending bounds only actual waiters.
        if self.waiting.len() >= self.config.max_pending {
            let count = self.waiting.len();
            self.waiting.retain(|r| !r.stream.is_closed());
            self.counters.cancelled += (count - self.waiting.len()) as u64;
        }
        ensure!(
            self.waiting.len() < self.config.max_pending,
            "request queue is full"
        );
        Ok(())
    }

    fn enqueue(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: u32,
        ignore_eos: bool,
        stream: Output,
    ) -> Result<u64> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| anyhow!("request ID exhausted"))?;
        self.waiting.push_back(Request {
            id,
            submitted: Instant::now(),
            prompt,
            max_new_tokens,
            ignore_eos,
            stream,
            cached: 0,
            generated: 0,
            next_input: 0,
        });
        self.admit();
        Ok(id)
    }

    fn admit(&mut self) {
        while !self.free.is_empty() {
            let Some(request) = self.waiting.pop_front() else {
                break;
            };
            if request.stream.is_closed() {
                self.counters.cancelled += 1;
                continue;
            }
            let slot = self.free.pop().expect("available KV slot");
            self.counters.admitted += 1;
            self.counters.total_queue_wait_us += request.submitted.elapsed().as_micros();
            self.slots[slot] = Some(request);
            self.prefill.push_back(slot);
        }
    }

    pub fn has_work(&self) -> bool {
        !self.prefill.is_empty() || !self.decode.is_empty() || !self.waiting.is_empty()
    }

    pub fn stats(&self) -> Stats {
        Stats {
            waiting: self.waiting.len(),
            active: self.config.slots - self.free.len(),
            ..self.counters.clone()
        }
    }

    /// Called by the same owner between steps; receiver drops are also observed automatically.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(i) = self.waiting.iter().position(|r| r.id == id) {
            self.waiting.remove(i);
        } else if let Some(slot) = self
            .slots
            .iter()
            .position(|r| r.as_ref().is_some_and(|r| r.id == id))
        {
            self.prefill.retain(|&s| s != slot);
            self.decode.retain(|&s| s != slot);
            self.release(slot);
        } else {
            return false;
        }
        self.counters.cancelled += 1;
        self.admit();
        true
    }

    fn release(&mut self, slot: usize) {
        self.slots[slot] = None;
        self.free.push(slot);
    }

    fn prepare(&mut self, phase: Phase) {
        let (limit, width, ready) = match phase {
            Phase::Prefill => (
                self.config.prefill_batch,
                self.config.prefill_chunk,
                &mut self.prefill,
            ),
            Phase::Decode => (self.config.decode_batch, 1, &mut self.decode),
        };
        let batch = &mut self.batch;
        batch.phase = phase;
        batch.width = width;
        batch.slots.clear();
        batch.positions.clear();
        batch.lengths.clear();
        batch.tokens.clear();
        while batch.slots.len() < limit {
            let Some(slot) = ready.pop_front() else { break };
            let r = self.slots[slot]
                .as_ref()
                .expect("scheduled slot owns request");
            if r.stream.is_closed() {
                self.slots[slot] = None;
                self.free.push(slot);
                self.counters.cancelled += 1;
                continue;
            }
            let tokens = match phase {
                Phase::Prefill => &r.prompt[r.cached..r.prompt.len().min(r.cached + width)],
                Phase::Decode => std::slice::from_ref(&r.next_input),
            };
            batch.positions.push(r.cached as u32);
            batch.slots.push(slot as u32);
            batch.lengths.push(tokens.len() as u32);
            batch.tokens.extend_from_slice(tokens);
            batch.tokens.resize(
                batch.tokens.len() + width - tokens.len(),
                self.config.pad_token,
            );
        }
    }

    fn poison(&mut self, message: String) {
        self.failure = Some(message.clone());
        for mut r in self
            .waiting
            .drain(..)
            .chain(self.slots.iter_mut().filter_map(Option::take))
        {
            let _ = r.stream.send(Event::Failed(message.clone()));
            self.counters.failed += 1;
        }
        self.free.clear();
        self.free.extend((0..self.config.slots).rev());
        self.prefill.clear();
        self.decode.clear();
    }

    pub fn tick(&mut self, executor: &mut impl Executor) -> Result<()> {
        if let Some(message) = &self.failure {
            return Err(anyhow!(message.clone()));
        }
        self.admit();
        let phase = if self.decode.is_empty()
            || (!self.prefill.is_empty() && self.last_phase == Phase::Decode)
        {
            Phase::Prefill
        } else {
            Phase::Decode
        };
        self.prepare(phase);
        let rows = self.batch.slots.len();
        if rows == 0 {
            return Ok(());
        }
        let started = Instant::now();
        let out = &mut self.output[..rows];
        // Detect a backend that omits an output row instead of re-emitting a previous token.
        out.fill(u32::MAX);
        let result = executor.step(&self.batch, out).and_then(|()| {
            ensure!(
                out.iter().all(|&id| id < self.config.vocab_size),
                "invalid next token from device"
            );
            Ok(())
        });
        if let Err(error) = result {
            self.poison(format!("{phase:?} execution failed: {error:#}"));
            return Err(error);
        }
        let completed = Instant::now();
        let micros = completed.duration_since(started).as_micros();
        let tokens = self
            .batch
            .lengths
            .iter()
            .map(|&n| u64::from(n))
            .sum::<u64>();
        match phase {
            Phase::Prefill => {
                self.counters.prefill_steps += 1;
                self.counters.prefill_rows += rows as u64;
                self.counters.prefill_input_tokens += tokens;
                self.counters.prefill_row_padding_tokens += self.batch.tokens.len() as u64 - tokens;
                self.counters.prefill_step_us += micros;
            }
            Phase::Decode => {
                self.counters.decode_steps += 1;
                self.counters.decode_input_tokens += tokens;
                self.counters.decode_step_us += micros;
            }
        }
        self.last_phase = phase;
        for row in 0..rows {
            let slot = self.batch.slots[row] as usize;
            let r = self.slots[slot]
                .as_mut()
                .expect("executed slot owns request");
            r.cached += self.batch.lengths[row] as usize;
            if phase == Phase::Prefill && r.cached < r.prompt.len() {
                self.prefill.push_back(slot);
                continue;
            }
            if phase == Phase::Prefill {
                // Host prompt storage is no longer needed once the complete prefix is cached.
                r.prompt = Vec::new();
            }
            let token = self.output[row];
            r.next_input = token;
            r.generated += 1;
            self.counters.generated_tokens += 1;
            let finish = if !r.ignore_eos && self.config.eos_tokens.contains(&token) {
                Some(FinishReason::Stop)
            } else if r.generated == r.max_new_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };
            if r.stream.send(Event::Token { token, finish }).is_err() {
                self.counters.cancelled += 1;
                self.release(slot);
                continue;
            }
            if r.generated == 1 {
                self.counters.first_tokens += 1;
                self.counters.total_first_token_us +=
                    completed.duration_since(r.submitted).as_micros();
            }
            if finish.is_some() {
                self.counters.finished += 1;
                self.release(slot);
            } else {
                self.decode.push_back(slot);
            }
        }
        self.admit();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    fn config() -> Config {
        Config {
            slots: 2,
            prefill_batch: 2,
            decode_batch: 1,
            prefill_chunk: 3,
            context_len: 16,
            vocab_size: 32,
            eos_tokens: vec![19],
            pad_token: 0,
            max_pending: 2,
        }
    }

    struct TraceExecutor {
        cached: Vec<Vec<u32>>,
        phases: Vec<Phase>,
        rows: Vec<(u32, u32, Vec<u32>)>,
    }

    impl TraceExecutor {
        fn new() -> Self {
            Self {
                cached: vec![Vec::new(), Vec::new()],
                phases: Vec::new(),
                rows: Vec::new(),
            }
        }
    }

    impl Executor for TraceExecutor {
        fn step(&mut self, batch: &Batch, out: &mut [u32]) -> Result<()> {
            self.phases.push(batch.phase);
            assert_eq!(batch.tokens.len(), batch.width * out.len());
            for (row, next) in out.iter_mut().enumerate() {
                let slot = batch.slots[row];
                let cache = &mut self.cached[slot as usize];
                let position = batch.positions[row];
                if position == 0 {
                    cache.clear();
                }
                assert_eq!(position as usize, cache.len());
                let n = batch.lengths[row] as usize;
                let input = &batch.tokens[row * batch.width..(row + 1) * batch.width];
                assert!(input[n..].iter().all(|&x| x == 0));
                self.rows.push((slot, position, input[..n].to_vec()));
                cache.extend_from_slice(&input[..n]);
                *next = (cache.last().unwrap() + 1) % 32;
            }
            Ok(())
        }
    }

    fn submit(
        s: &mut Scheduler,
        prompt: &[u32],
        max: u32,
        ignore_eos: bool,
    ) -> (u64, UnboundedReceiver<Event>) {
        let (tx, rx) = unbounded_channel();
        (s.submit(prompt.to_vec(), max, ignore_eos, tx).unwrap(), rx)
    }

    fn drained(rx: &mut UnboundedReceiver<Event>) -> Vec<(u32, Option<FinishReason>)> {
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                Event::Token { token, finish } => events.push((token, finish)),
                Event::Failed(e) => panic!("unexpected failure: {e}"),
            }
        }
        events
    }

    #[test]
    fn chunk_tails_and_dynamic_admission_preserve_request_order_and_state() {
        let mut s = Scheduler::new(config()).unwrap();
        let mut dev = TraceExecutor::new();
        let (_, mut a) = submit(&mut s, &[6, 7, 8, 9, 10], 3, true);
        s.tick(&mut dev).unwrap();
        assert!(a.try_recv().is_err(), "partial prefill cannot emit a token");
        let (_, mut b) = submit(&mut s, &[17], 8, false);
        s.tick(&mut dev).unwrap();
        let (_, mut c) = submit(&mut s, &[2], 1, false);
        while s.has_work() {
            s.tick(&mut dev).unwrap();
        }
        assert_eq!(
            drained(&mut a),
            [(11, None), (12, None), (13, Some(FinishReason::Length))]
        );
        assert_eq!(
            drained(&mut b),
            [(18, None), (19, Some(FinishReason::Stop))]
        );
        assert_eq!(drained(&mut c), [(3, Some(FinishReason::Length))]);
        assert_eq!(
            dev.rows,
            [
                (0, 0, vec![6, 7, 8]),
                (0, 3, vec![9, 10]),
                (1, 0, vec![17]),
                (0, 5, vec![11]),
                (1, 1, vec![18]),
                (1, 0, vec![2]),
                (0, 6, vec![12]),
            ]
        );
        let stats = s.stats();
        assert_eq!(
            (
                stats.finished,
                stats.active,
                stats.waiting,
                stats.generated_tokens
            ),
            (3, 0, 0, 6)
        );
        assert_eq!(
            (stats.prefill_input_tokens, stats.decode_input_tokens),
            (7, 3)
        );
        assert_eq!(
            (
                stats.prefill_rows,
                stats.prefill_row_padding_tokens,
                stats.first_tokens
            ),
            (4, 5, 3)
        );
    }

    #[test]
    fn decode_advances_while_a_long_prefill_is_chunked() {
        let mut s = Scheduler::new(config()).unwrap();
        let mut dev = TraceExecutor::new();
        let (_, mut a) = submit(&mut s, &[18], 6, true);
        s.tick(&mut dev).unwrap();
        let (_, mut b) = submit(&mut s, &[5; 12], 1, true);
        while s.has_work() {
            s.tick(&mut dev).unwrap();
        }
        assert_eq!(
            &dev.phases[..8],
            &[
                Phase::Prefill,
                Phase::Decode,
                Phase::Prefill,
                Phase::Decode,
                Phase::Prefill,
                Phase::Decode,
                Phase::Prefill,
                Phase::Decode
            ]
        );
        assert_eq!(
            drained(&mut a).len(),
            6,
            "ignore_eos=true must pass the stop ID"
        );
        assert_eq!(drained(&mut b), [(6, Some(FinishReason::Length))]);
    }

    #[test]
    fn cancellation_before_or_after_prefill_reclaims_slots_only_between_steps() {
        let mut cfg = config();
        cfg.slots = 1;
        let mut s = Scheduler::new(cfg).unwrap();
        let mut dev = TraceExecutor::new();
        let (old, first) = submit(&mut s, &[4], 4, true);
        s.tick(&mut dev).unwrap();
        let (queued, _second) = submit(&mut s, &[8], 2, false);
        let (_, mut third) = submit(&mut s, &[2], 1, true);
        assert!(s.cancel(queued));
        assert!(!s.cancel(queued));
        drop(first);
        while s.has_work() {
            s.tick(&mut dev).unwrap();
        }
        assert!(!s.cancel(old));
        assert_eq!(dev.rows, [(0, 0, vec![4]), (0, 0, vec![2])]);
        assert_eq!(drained(&mut third), [(3, Some(FinishReason::Length))]);
        assert_eq!(s.stats().cancelled, 2);
    }

    #[test]
    fn client_can_disconnect_and_enqueue_a_replacement_during_device_execution() {
        use std::sync::mpsc::{Receiver, Sender, channel};
        use std::time::Duration;

        struct InFlight {
            started: Sender<()>,
            replaced: Receiver<()>,
            device: TraceExecutor,
        }
        impl Executor for InFlight {
            fn step(&mut self, batch: &Batch, out: &mut [u32]) -> Result<()> {
                self.started.send(())?;
                self.replaced.recv_timeout(Duration::from_secs(5))?;
                self.device.step(batch, out)
            }
        }
        let mut cfg = config();
        cfg.slots = 1;
        let mut s = Scheduler::new(cfg).unwrap();
        let (_, old_stream) = submit(&mut s, &[10], 4, true);
        let (started, start) = channel();
        let (done, replaced) = channel();
        let (ingress, incoming) = channel();
        let (new_sender, mut new_stream) = unbounded_channel();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                start.recv_timeout(Duration::from_secs(5)).unwrap();
                drop(old_stream);
                ingress.send((vec![20], new_sender)).unwrap();
                done.send(()).unwrap();
            });
            let mut flight = InFlight {
                started,
                replaced,
                device: TraceExecutor::new(),
            };
            s.tick(&mut flight).unwrap();
            let (prompt, stream) = incoming.recv_timeout(Duration::from_secs(5)).unwrap();
            s.submit(prompt, 1, false, stream).unwrap();
            s.tick(&mut flight.device).unwrap();
            assert_eq!(flight.device.rows, [(0, 0, vec![10]), (0, 0, vec![20])]);
        });
        assert_eq!(drained(&mut new_stream), [(21, Some(FinishReason::Length))]);
        assert_eq!((s.stats().cancelled, s.stats().first_tokens), (1, 1));
    }

    #[test]
    fn dispatch_error_or_missing_row_poisons_all_slots_and_waiters() {
        struct Broken(bool);
        impl Executor for Broken {
            fn step(&mut self, _batch: &Batch, out: &mut [u32]) -> Result<()> {
                out[0] = 5;
                ensure!(!self.0, "rank failed after writing its cache");
                Ok(()) // Unwritten second row must also poison the batch.
            }
        }
        for dispatch_error in [true, false] {
            let mut s = Scheduler::new(config()).unwrap();
            let mut streams: Vec<_> = (0..3).map(|_| submit(&mut s, &[2], 1, false).1).collect();
            let mut dev = Broken(dispatch_error);
            assert!(s.tick(&mut dev).is_err());
            for rx in &mut streams {
                assert!(matches!(rx.try_recv(), Ok(Event::Failed(_))));
            }
            assert!(!s.has_work());
            assert_eq!(s.stats().failed, 3);
            let (tx, _rx) = unbounded_channel();
            assert!(s.submit(vec![2], 1, false, tx).is_err());
            assert!(s.tick(&mut dev).is_err());
        }
    }

    #[test]
    fn validates_request_and_bounds_the_wait_queue() {
        let mut cfg = config();
        cfg.slots = 1;
        cfg.max_pending = 1;
        let mut s = Scheduler::new(cfg).unwrap();
        let (tx, _rx) = unbounded_channel();
        for (tokens, n) in [(vec![], 1), (vec![3], 0), (vec![32], 1), (vec![1; 16], 1)] {
            assert!(s.submit(tokens, n, false, tx.clone()).is_err());
        }
        let (active, _first) = submit(&mut s, &[1], 2, true);
        let (_pending, second) = submit(&mut s, &[1], 2, true);
        assert!(s.submit(vec![2], 2, true, tx.clone()).is_err());
        drop(second);
        s.submit(vec![2], 2, true, tx).unwrap();
        assert!(s.cancel(active));
        assert_eq!(
            (s.stats().cancelled, s.stats().active, s.stats().waiting),
            (2, 1, 0)
        );
    }

    #[test]
    fn reserved_stream_disconnect_reclaims_slot_and_delivers_waiter() {
        let mut cfg = config();
        cfg.slots = 1;
        let mut s = Scheduler::new(cfg).unwrap();
        let mut device = TraceExecutor::new();
        let (_, abandoned) = s.submit_stream(vec![1], 4, true).unwrap();
        let (_, mut next) = s.submit_stream(vec![18], 4, false).unwrap();
        s.tick(&mut device).unwrap();
        drop(abandoned);
        while s.has_work() {
            s.tick(&mut device).unwrap();
        }
        assert_eq!(
            next.try_recv(),
            Ok(Event::Token {
                token: 19,
                finish: Some(FinishReason::Stop)
            })
        );
        assert_eq!(
            next.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        );
        assert_eq!((s.stats().finished, s.stats().cancelled), (1, 1));
    }
}
