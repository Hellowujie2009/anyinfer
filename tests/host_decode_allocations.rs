//! Counts only this test thread's Rust allocator calls. The production binary
//! and its latency measurements use their normal allocator without these hooks.
use anyinfer::native::scheduler::{Batch, Config, Event, Executor, FinishReason, Scheduler};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Calls {
    allocate: u64,
    reallocate: u64,
    free: u64,
}

thread_local! {
    static CALLS: Cell<Option<Calls>> = const { Cell::new(None) };
}

fn record(update: impl FnOnce(&mut Calls)) {
    let _ = CALLS.try_with(|counter| {
        if let Some(mut snapshot) = counter.get() {
            update(&mut snapshot);
            counter.set(Some(snapshot));
        }
    });
}

struct CountingAllocator;

// SAFETY: every operation forwards its unchanged allocation arguments to
// System. The counter uses const-initialized TLS and performs no allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(|c| c.allocate += 1);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(|c| c.allocate += 1);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(|c| c.reallocate += 1);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(|c| c.free += 1);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct FixedOutput;

impl Executor for FixedOutput {
    fn step(&mut self, _: &Batch, next: &mut [u32]) -> anyhow::Result<()> {
        next.fill(2);
        Ok(())
    }
}

#[test]
fn steady_decode_and_delivery_do_not_call_the_rust_allocator() {
    const STEPS: u32 = 4096;
    let mut scheduler = Scheduler::new(Config {
        slots: 1,
        prefill_batch: 1,
        decode_batch: 1,
        prefill_chunk: 4,
        context_len: STEPS as usize + 4,
        vocab_size: 8,
        eos_tokens: vec![0],
        pad_token: 0,
        max_pending: 1,
    })
    .unwrap();
    let (_, mut tokens) = scheduler.submit_stream(vec![1], STEPS + 3, true).unwrap();
    let mut executor = FixedOutput;
    let expected = Event::Token {
        token: 2,
        finish: None,
    };
    for _ in 0..2 {
        scheduler.tick(&mut executor).unwrap();
        assert_eq!(tokens.try_recv().unwrap(), expected);
    }
    CALLS.set(Some(Calls::default()));
    for _ in 0..STEPS {
        scheduler.tick(&mut executor).unwrap();
        assert_eq!(tokens.try_recv().unwrap(), expected);
    }
    let observed = CALLS.replace(None).unwrap();
    assert_eq!(observed, Calls::default());
    scheduler.tick(&mut executor).unwrap();
    assert_eq!(
        tokens.try_recv().unwrap(),
        Event::Token {
            token: 2,
            finish: Some(FinishReason::Length)
        }
    );
    assert!(!scheduler.has_work());
}
