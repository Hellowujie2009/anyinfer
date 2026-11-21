use super::scheduler::{Event, FinishReason};
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use tokio::sync::{Notify, mpsc::error::TryRecvError};

// The upper word of published describes the boundary after the last published token.
const OPEN: u64 = 0;
const LENGTH: u64 = 1;
const STOP: u64 = 2;
const FAILED: u64 = 3;
const CLOSED: u64 = 4;

struct Shared {
    tokens: Vec<AtomicU32>,
    published: AtomicU64,
    disconnected: AtomicBool,
    failure: OnceLock<String>,
    ready: Notify,
}

pub(crate) struct Writer {
    shared: Arc<Shared>,
    written: u32,
    done: bool,
}

/// Single-consumer token stream. The payload reserves four bytes per requested
/// output token on submission. A slow consumer can read through completion
/// without delaying the device owner. Drop the receiver to cancel or release
/// its storage once the scheduler has relinquished its own reference.
pub struct TokenStream {
    shared: Arc<Shared>,
    read: u32,
    failure_seen: bool,
}

pub(crate) fn channel(limit: u32) -> Result<(Writer, TokenStream)> {
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(limit as usize)
        .context("reserve request output tokens")?;
    tokens.resize_with(limit as usize, AtomicU32::default);
    let shared = Arc::new(Shared {
        tokens,
        published: AtomicU64::new(0),
        disconnected: AtomicBool::new(false),
        failure: OnceLock::new(),
        ready: Notify::new(),
    });
    Ok((
        Writer {
            shared: shared.clone(),
            written: 0,
            done: false,
        },
        TokenStream {
            shared,
            read: 0,
            failure_seen: false,
        },
    ))
}

impl Writer {
    pub(crate) fn is_closed(&self) -> bool {
        self.shared.disconnected.load(Ordering::Acquire)
    }

    fn publish(&mut self, state: u64) {
        self.done = state != OPEN;
        // One release publishes the payload and its final reason together. A
        // receiver can never observe the last token without its finish marker.
        self.shared
            .published
            .store((state << 32) | u64::from(self.written), Ordering::Release);
        self.shared.ready.notify_one();
    }

    pub(crate) fn send(&mut self, event: Event) -> Result<(), ()> {
        if self.is_closed() {
            return Err(());
        }
        let state = match event {
            Event::Token { token, finish } => {
                self.shared.tokens[self.written as usize].store(token, Ordering::Relaxed);
                self.written += 1;
                match finish {
                    None => OPEN,
                    Some(FinishReason::Length) => LENGTH,
                    Some(FinishReason::Stop) => STOP,
                }
            }
            Event::Failed(error) => {
                self.shared
                    .failure
                    .set(error)
                    .expect("only one terminal error");
                FAILED
            }
        };
        self.publish(state);
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.done {
            self.publish(CLOSED);
        }
    }
}

impl TokenStream {
    /// Returns Disconnected after all buffered events have been observed.
    pub fn try_recv(&mut self) -> Result<Event, TryRecvError> {
        let published = self.shared.published.load(Ordering::Acquire);
        let count = published as u32;
        let state = published >> 32;
        if self.read < count {
            let token = self.shared.tokens[self.read as usize].load(Ordering::Relaxed);
            self.read += 1;
            let finish = match (self.read == count, state) {
                (true, LENGTH) => Some(FinishReason::Length),
                (true, STOP) => Some(FinishReason::Stop),
                _ => None,
            };
            return Ok(Event::Token { token, finish });
        }
        if state == FAILED && !self.failure_seen {
            self.failure_seen = true;
            return Ok(Event::Failed(
                self.shared.failure.get().expect("published error").clone(),
            ));
        }
        Err(if state == OPEN {
            TryRecvError::Empty
        } else {
            TryRecvError::Disconnected
        })
    }

    /// Waits for one event, or returns None when the sender is closed and the
    /// buffered events are exhausted. Cancelling this future consumes no token.
    pub async fn recv(&mut self) -> Option<Event> {
        loop {
            match self.try_recv() {
                Ok(event) => return Some(event),
                Err(TryRecvError::Disconnected) => return None,
                // notify_one retains a permit even if the writer races this await.
                Err(TryRecvError::Empty) => self.shared.ready.notified().await,
            }
        }
    }
}

impl Drop for TokenStream {
    fn drop(&mut self) {
        self.shared.disconnected.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::task::{Context, Poll, Wake, Waker};

    fn token(token: u32, finish: Option<FinishReason>) -> Event {
        Event::Token { token, finish }
    }

    struct Signal(AtomicBool);

    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn waiting_and_cancelled_futures_preserve_notifications_and_close() {
        let (mut writer, mut rx) = channel(2).unwrap();
        let signal = Arc::new(Signal(AtomicBool::new(false)));
        let waker = Waker::from(signal.clone());
        let mut cx = Context::from_waker(&waker);
        {
            let mut pending = std::pin::pin!(rx.recv());
            assert!(pending.as_mut().poll(&mut cx).is_pending());
            writer.send(token(3, None)).unwrap();
            assert!(signal.0.load(Ordering::Relaxed));
            // Cancel while the woken future has yet to consume the value.
        }
        assert_eq!(rx.try_recv(), Ok(token(3, None)));
        let mut waiting = std::pin::pin!(rx.recv());
        assert!(waiting.as_mut().poll(&mut cx).is_pending());
        signal.0.store(false, Ordering::Relaxed);
        drop(writer);
        assert!(signal.0.load(Ordering::Relaxed));
        assert_eq!(waiting.poll(&mut cx), Poll::Ready(None));
    }

    #[test]
    fn concurrent_reader_sees_every_token_once_including_final_reason() {
        const N: u32 = 2048;
        let (mut writer, mut rx) = channel(N).unwrap();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                for i in 0..N {
                    writer
                        .send(token(i, (i + 1 == N).then_some(FinishReason::Length)))
                        .unwrap();
                }
            });
            for i in 0..N {
                let event = loop {
                    match rx.try_recv() {
                        Err(TryRecvError::Empty) => std::thread::yield_now(),
                        result => break result.unwrap(),
                    }
                };
                assert_eq!(
                    event,
                    token(i, (i + 1 == N).then_some(FinishReason::Length))
                );
            }
            assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
        });
    }

    #[test]
    fn slow_reader_drains_published_tokens_before_error_or_eos() {
        for event in [
            Event::Failed("device error".into()),
            token(4, Some(FinishReason::Stop)),
        ] {
            let (mut writer, mut rx) = channel(8).unwrap();
            writer.send(token(3, None)).unwrap();
            writer
                .send(match &event {
                    Event::Failed(e) => Event::Failed(e.clone()),
                    Event::Token { token: id, finish } => token(*id, *finish),
                })
                .unwrap();
            drop(writer);
            assert_eq!(rx.try_recv(), Ok(token(3, None)));
            assert_eq!(rx.try_recv(), Ok(event));
            assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
        }
    }
}
