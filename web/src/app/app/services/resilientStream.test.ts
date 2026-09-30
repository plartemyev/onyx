/** @jest-environment jsdom */
import type { PacketType } from "./lib";
import {
  isStreamResyncMarker,
  resilientPacketStream,
  StreamHttpError,
  type ResilientStreamOptions,
} from "./resilientStream";

type FakePacket = Record<string, unknown>;
type OutputStream = AsyncGenerator<PacketType, void, unknown>;

const heartbeat: FakePacket = { obj: { type: "chat_heartbeat" } };
const packet = (n: number): FakePacket => ({
  obj: { type: "message_start", seq: n },
});

/** Async generator over fixed items, throwing after `throwAfterCount` yields. */
function streamOf(items: FakePacket[], throwAfterCount?: number): OutputStream {
  const bound =
    throwAfterCount === undefined
      ? items.length
      : Math.max(items.length, throwAfterCount + 1);
  return (async function* () {
    for (let i = 0; i < bound; i++) {
      if (throwAfterCount !== undefined && i >= throwAfterCount) {
        throw new Error("transport dropped");
      }
      if (i < items.length) {
        yield items[i] as unknown as PacketType;
      }
    }
  })() as unknown as OutputStream;
}

async function collect(
  generator: AsyncGenerator<PacketType | { resync: true }, void, unknown>
): Promise<Array<FakePacket | "RESYNC">> {
  const out: Array<FakePacket | "RESYNC"> = [];
  for await (const item of generator) {
    if (isStreamResyncMarker(item)) {
      out.push("RESYNC");
    } else {
      out.push(item as unknown as FakePacket);
    }
  }
  return out;
}

const idleProbe = () => Promise.resolve({ processing: false, run_id: null });

function buildOptions(
  overrides: Partial<ResilientStreamOptions>
): ResilientStreamOptions {
  return {
    resumeConnect: jest.fn(),
    probe: idleProbe,
    ...overrides,
  };
}

describe("resilientPacketStream", () => {
  it("forwards a clean live stream and filters heartbeats", async () => {
    const items = [packet(1), heartbeat, packet(2)];
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () => Promise.resolve(streamOf(items)),
        resumeConnect: () => Promise.resolve(streamOf([])),
      })
    );

    await expect(collect(generator)).resolves.toEqual([packet(1), packet(2)]);
  });

  it("re-attaches with a resync marker after a mid-stream drop while the run is alive", async () => {
    const resumeConnect = jest
      .fn()
      .mockImplementation(() =>
        Promise.resolve(streamOf([packet(1), packet(2), packet(3)]))
      );
    const onSessionRefresh = jest.fn();
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () => Promise.resolve(streamOf([packet(1)], 1)),
        resumeConnect: () => resumeConnect(new AbortController().signal),
        probe: () => Promise.resolve({ processing: true, run_id: 7 }),
        onSessionRefresh,
      })
    );

    // The live drop yields packet(1); the replay re-delivers everything.
    await expect(collect(generator)).resolves.toEqual([
      packet(1),
      "RESYNC",
      packet(1),
      packet(2),
      packet(3),
    ]);
    expect(resumeConnect).toHaveBeenCalledTimes(1);
    expect(onSessionRefresh).not.toHaveBeenCalled();
  });

  it("reports a session refresh when the run finished while disconnected", async () => {
    const onSessionRefresh = jest.fn();
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () => Promise.resolve(streamOf([packet(1)], 1)),
        resumeConnect: () => Promise.reject(new Error("offline")),
        probe: () => Promise.resolve({ processing: false, run_id: null }),
        onSessionRefresh,
      })
    );

    await expect(collect(generator)).resolves.toEqual([packet(1)]);
    expect(onSessionRefresh).toHaveBeenCalledTimes(1);
  });

  it("watches the fence when the buffer is gapped and refreshes once the run ends", async () => {
    const onSessionRefresh = jest.fn();
    const probe = jest
      .fn()
      .mockResolvedValueOnce({ processing: true, run_id: 7 })
      .mockResolvedValueOnce({ processing: false, run_id: null });
    const gapStream = (async function* () {
      yield packet(2) as unknown as PacketType;
      yield { buffer_gap: true } as unknown as PacketType;
    })() as unknown as OutputStream;
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () => Promise.resolve(streamOf([packet(1)], 1)),
        resumeConnect: () => Promise.resolve(gapStream),
        probe,
        onSessionRefresh,
        watchPollMs: 5,
      })
    );

    // The gapped replay delivers what survived, then the client watches the
    // fence until the run ends and refreshes from the persisted session.
    await expect(collect(generator)).resolves.toEqual([
      packet(1),
      "RESYNC",
      packet(2),
    ]);
    expect(onSessionRefresh).toHaveBeenCalledTimes(1);
  });

  it("throws when the send never got through and nothing is running", async () => {
    const onSessionRefresh = jest.fn();
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () => Promise.reject(new Error("offline")),
        resumeConnect: () => Promise.reject(new Error("offline")),
        probe: () => Promise.resolve({ processing: false, run_id: null }),
        onSessionRefresh,
      })
    );

    await expect(collect(generator)).rejects.toThrow(
      /did not accept this message/
    );
    expect(onSessionRefresh).not.toHaveBeenCalled();
  });

  it("propagates fatal 4xx send errors instead of retrying", async () => {
    const resumeConnect = jest
      .fn()
      .mockRejectedValue(new Error("should not resume"));
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: () =>
          Promise.reject(new StreamHttpError(400, "bad request")),
        resumeConnect: () => resumeConnect(new AbortController().signal),
      })
    );

    await expect(collect(generator)).rejects.toThrow("bad request");
    expect(resumeConnect).not.toHaveBeenCalled();
  });

  it("ends silently when the consumer aborts", async () => {
    const controller = new AbortController();
    const generator = resilientPacketStream(
      buildOptions({
        sendConnect: (signal) =>
          Promise.resolve(
            (async function* (): AsyncGenerator<PacketType, void, unknown> {
              yield packet(1) as unknown as PacketType;
              // Simulates the fetch dying on user abort.
              while (!signal.aborted) {
                await new Promise((resolve) => setTimeout(resolve, 5));
              }
              throw new Error("AbortError");
            })() as unknown as OutputStream
          ),
        resumeConnect: () => Promise.reject(new Error("should not resume")),
        signal: controller.signal,
      })
    );

    const collected = collect(generator);
    await new Promise((resolve) => setTimeout(resolve, 20));
    controller.abort();
    await expect(collected).resolves.toEqual([packet(1)]);
  });
});
