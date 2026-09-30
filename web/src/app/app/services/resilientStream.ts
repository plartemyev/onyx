import type { PacketType } from "./lib";

/**
 * Transport resilience for chat streams.
 *
 * The backend keeps draining a run into the shared stream buffer after the
 * client disconnects, and `GET /chat-session/{id}/resume-stream` replays +
 * tails that buffer. This wrapper turns any transport failure of the live
 * send-stream into a re-attach to that buffer instead of a dead end:
 *
 * - idle watchdog: heartbeats arrive every ~15s on both the live and the
 *   resumed path, so total silence for 90s means the connection is dead
 *   (half-open TCP on mobile networks never delivers an error event).
 * - reconnect loop: capped backoff, woken early by `online` and
 *   `visibilitychange`; each attempt replays the buffer from the beginning.
 * - markers: `StreamResyncMarker` tells the consumer to rebuild its
 *   accumulated stream state from the replay; the session-refresh outcome is
 *   reported via `onSessionRefresh` when the probe says the run is over.
 *
 * Reconnecting never re-POSTs the message — that would start a second run.
 */

/** Idle time before a silently-dead connection is force-closed for re-attach. */
const STREAM_IDLE_TIMEOUT_MS = 90_000;
const PROBE_TIMEOUT_MS = 15_000;
const RECONNECT_BACKOFF_MS = [1_000, 2_000, 5_000, 10_000, 15_000, 30_000];
/** Watch-mode cadence while a live run's buffer can no longer be replayed. */
const WATCH_POLL_INTERVAL_MS = 10_000;

/** Emitted before a replay from the buffer after a transport failure. */
export interface StreamResyncMarker {
  readonly resync: true;
}

const RESYNC_MARKER: StreamResyncMarker = { resync: true };

export function isStreamResyncMarker(
  packet: PacketType | StreamResyncMarker
): packet is StreamResyncMarker {
  // No wire packet carries a top-level `resync` field.
  return "resync" in packet;
}

function isHeartbeatPacket(packet: PacketType | StreamResyncMarker): boolean {
  return "obj" in packet && packet.obj?.type === "chat_heartbeat";
}

/** Error with an HTTP status attached, thrown by the stream fetchers. */
export class StreamHttpError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** 4xx (except 408) from the initial send: retrying cannot help. */
function isFatalSendError(error: unknown): boolean {
  return (
    error instanceof StreamHttpError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408
  );
}

export interface ProcessingProbeResult {
  processing: boolean;
  run_id: number | null;
}

/**
 * Marker the resume endpoint emits when the stream buffer can no longer be
 * replayed (chunk evicted/expired). The run itself may still be alive, so a
 * gap must switch the client to fence-watch mode, not to a completed state.
 */
export interface BufferGapMarker {
  buffer_gap: true;
}

export interface ResilientStreamOptions {
  /** Starts the live run. Omit for tail-only consumers (resume attach). */
  sendConnect?: (
    signal: AbortSignal
  ) => Promise<AsyncGenerator<PacketType, void, unknown>>;
  /** Re-attaches to the run's buffered stream from the beginning. */
  resumeConnect: (
    signal: AbortSignal
  ) => Promise<AsyncGenerator<PacketType, void, unknown>>;
  /** Cheap fence probe deciding between re-attach and persisted-state refresh. */
  probe: () => Promise<ProcessingProbeResult | null>;
  signal?: AbortSignal;
  /** Fires once per drop streak, before the first re-attach attempt. */
  onReconnectStart?: () => void;
  /** The run ended while the client was disconnected: refetch the session. */
  onSessionRefresh?: () => void;
  /** Watch-mode fence poll cadence (tests shrink the default 10s). */
  watchPollMs?: number;
}

type NextResult = IteratorResult<PacketType, void>;

/** One connection attempt: fresh abort channel + idle watchdog. */
function createAttempt(signal?: AbortSignal) {
  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", onOuterAbort, { once: true });
    }
  }

  /** Rejects when idle too long, killing the underlying fetch. */
  const nextWithWatchdog = (
    iterator: AsyncIterator<PacketType, void, unknown>
  ): Promise<NextResult> =>
    new Promise<NextResult>((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout> | null = null;
      const onTimeout = () => {
        timer = null;
        controller.abort();
        reject(new Error("chat stream idle timeout"));
      };
      timer = setTimeout(onTimeout, STREAM_IDLE_TIMEOUT_MS);
      iterator
        .next()
        .then((value) => {
          if (timer !== null) {
            clearTimeout(timer);
          }
          resolve(value);
        })
        .catch((error: unknown) => {
          if (timer !== null) {
            clearTimeout(timer);
          }
          reject(error);
        });
    });

  const dispose = () => {
    signal?.removeEventListener("abort", onOuterAbort);
    controller.abort();
  };

  return { controller, nextWithWatchdog, dispose };
}

/**
 * Drives the stream connections with a watchdog and reconnects through
 * `resumeConnect` on any transport failure. Yields application packets plus
 * resync markers; heartbeats are consumed as watchdog activity and never
 * yielded.
 */
export async function* resilientPacketStream(
  options: ResilientStreamOptions
): AsyncGenerator<PacketType | StreamResyncMarker, void, unknown> {
  // Backoff is interrupted by connectivity return or the tab becoming
  // visible: the moment either happens, re-attach instead of waiting out
  // the timer.
  let wake: (() => void) | null = null;
  const listenerCleanup = new AbortController();
  window.addEventListener(
    "online",
    () => {
      wake?.();
    },
    { signal: listenerCleanup.signal }
  );
  document.addEventListener(
    "visibilitychange",
    () => {
      if (document.visibilityState === "visible") {
        wake?.();
      }
    },
    { signal: listenerCleanup.signal }
  );

  const interruptibleDelay = (ms: number): Promise<boolean> =>
    new Promise((resolve) => {
      if (options.signal?.aborted) {
        resolve(false);
        return;
      }
      const timer = setTimeout(() => {
        wake = null;
        resolve(true);
      }, ms);
      wake = () => {
        clearTimeout(timer);
        wake = null;
        resolve(true);
      };
    });

  const probeOnce = (): Promise<ProcessingProbeResult | null> =>
    Promise.race([
      options.probe().catch(() => null),
      new Promise<null>((resolve) => {
        const timer = setTimeout(() => resolve(null), PROBE_TIMEOUT_MS);
        listenerCleanup.signal.addEventListener("abort", () => {
          clearTimeout(timer);
        });
      }),
    ]);

  let emittedAnyPacket = false;
  let sendConnectFailed = false;
  let reconnectAnnounced = false;
  let backoffIndex = 0;

  let attempt = createAttempt(options.signal);
  try {
    let firstAttempt = true;
    while (true) {
      if (options.signal?.aborted) {
        return;
      }
      const isSendAttempt = firstAttempt && options.sendConnect !== undefined;
      firstAttempt = false;
      let connectionDeliveredPackets = false;
      let bufferGap = false;
      try {
        const generator = await (isSendAttempt
          ? options.sendConnect!(attempt.controller.signal)
          : options.resumeConnect(attempt.controller.signal));
        if (!isSendAttempt && emittedAnyPacket) {
          // The replay supersedes everything already delivered.
          yield RESYNC_MARKER;
        }
        const iterator = generator[Symbol.asyncIterator]();
        while (true) {
          const result = await attempt.nextWithWatchdog(iterator);
          if (result.done) {
            break;
          }
          const rawPacket = result.value;
          if ("buffer_gap" in rawPacket) {
            // The replay cannot continue; the run may still be alive.
            bufferGap = true;
            break;
          }
          const packet = rawPacket;
          if (isHeartbeatPacket(packet)) {
            continue;
          }
          connectionDeliveredPackets = true;
          emittedAnyPacket = true;
          yield packet;
        }
        // A connection that ends without a gap marker reached the buffer's
        // done marker (or is the live response finishing with the run): the
        // run is over and everything deliverable was delivered.
        if (!bufferGap) {
          return;
        }
      } catch (error) {
        // The watchdog kill and transport failures land here; a user abort
        // surfaces through `options.signal` and is handled below.
        if (isSendAttempt) {
          sendConnectFailed = true;
          if (isFatalSendError(error)) {
            throw error;
          }
        }
      }

      if (options.signal?.aborted) {
        return;
      }

      // Backoff grows only across connections that delivered nothing — a
      // connection that streamed packets means the network itself is fine.
      backoffIndex = connectionDeliveredPackets ? 0 : backoffIndex + 1;
      const delayMs =
        RECONNECT_BACKOFF_MS[
          Math.min(backoffIndex, RECONNECT_BACKOFF_MS.length - 1)
        ]!;
      if (delayMs > 0) {
        if (!reconnectAnnounced) {
          reconnectAnnounced = true;
          options.onReconnectStart?.();
        }
        const completed = await interruptibleDelay(delayMs);
        if (!completed || options.signal?.aborted) {
          return;
        }
      }

      let probeResult: ProcessingProbeResult | null = null;
      try {
        probeResult = await probeOnce();
      } catch {
        probeResult = null;
      }
      if (options.signal?.aborted) {
        return;
      }
      if (probeResult && !probeResult.processing) {
        if (emittedAnyPacket || !sendConnectFailed) {
          // The run finished while we were away; the persisted session has
          // the authoritative outcome. Only trust this when the send itself
          // did not fail unreadably — otherwise the run may never have
          // started and the consumer must hear about the failure.
          options.onSessionRefresh?.();
          return;
        }
        throw new Error(
          "The chat service did not accept this message. Please try again."
        );
      }
      if (bufferGap && probeResult?.processing) {
        // Watch mode: the buffer is unrecoverable but the writer is alive.
        // Poll the fence until the run ends, then settle from the persisted
        // session. (Chunk eviction is permanent within a run, so replaying
        // the gapped buffer again would only hammer the cache.)
        while (true) {
          const completed = await interruptibleDelay(
            options.watchPollMs ?? WATCH_POLL_INTERVAL_MS
          );
          if (!completed || options.signal?.aborted) {
            return;
          }
          const watchProbe = await probeOnce();
          if (options.signal?.aborted) {
            return;
          }
          if (watchProbe === null) {
            // Probe unreachable (offline): keep watching.
            continue;
          }
          if (!watchProbe.processing) {
            options.onSessionRefresh?.();
            return;
          }
        }
      }
      // The run is still processing (or the probe is unreachable): loop and
      // re-attach to the buffer.
      attempt.dispose();
      attempt = createAttempt(options.signal);
    }
  } finally {
    listenerCleanup.abort();
    attempt.dispose();
  }
}
