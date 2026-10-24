/** @jest-environment jsdom */
import type { Packet } from "@/app/app/services/streamingModels";
import type { Message } from "@/app/app/interfaces";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";

const TERMINATED_PLACEHOLDER =
  "Response was terminated prior to completion, try regenerating.";

jest.mock("./lib", () => ({
  resumeStream: jest.fn(),
  getChatSessionProcessingStatus: jest.fn(),
  processRawChatHistory: jest.fn(),
}));

import {
  getChatSessionProcessingStatus,
  processRawChatHistory,
  resumeStream,
} from "./lib";
import { attachToRunningRun } from "./resumeRun";

const mockedResumeStream = jest.mocked(resumeStream);
const mockedProbe = jest.mocked(getChatSessionProcessingStatus);
const mockedProcessRawChatHistory = jest.mocked(processRawChatHistory);

const packet = (type: string): Packet =>
  ({ obj: { type } }) as unknown as Packet;

/** Resume stream the test drives by hand: push packets, then finish the run. */
function controlledResumeStream() {
  const queue: Packet[] = [];
  let notify: (() => void) | null = null;
  let finished = false;

  const generator = async function* (): AsyncGenerator<Packet, void, unknown> {
    while (true) {
      if (queue.length > 0) {
        yield queue.shift()!;
        continue;
      }
      if (finished) {
        return;
      }
      await new Promise<void>((resolve) => {
        notify = resolve;
      });
    }
  };

  return {
    generator,
    push(item: Packet) {
      queue.push(item);
      notify?.();
      notify = null;
    },
    finish() {
      finished = true;
      notify?.();
      notify = null;
    },
  };
}

function placeholderNode(nodeId: number): Message {
  return {
    nodeId,
    messageId: nodeId,
    message: TERMINATED_PLACEHOLDER,
    type: "assistant",
    files: [],
    toolCall: null,
    parentNodeId: null,
    // What get-chat-session synthesizes from the reserved row's text.
    packets: [packet("message_start"), packet("message_delta")],
    packetCount: 2,
  };
}

function seedSession(sessionId: string, node: Message) {
  useChatSessionStore
    .getState()
    .updateSessionAndMessageTree(sessionId, new Map([[node.nodeId, node]]));
}

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

const originalFetch = global.fetch;

afterEach(() => {
  jest.resetAllMocks();
  global.fetch = originalFetch;
});

describe("attachToRunningRun", () => {
  it("clears the terminated-response placeholder and rebuilds from the replay", async () => {
    const sessionId = "session-resume-1";
    const node = placeholderNode(42);
    seedSession(sessionId, node);

    const settledNode: Message = {
      ...placeholderNode(42),
      message: "Final answer",
      packets: [packet("message_start")],
      packetCount: 1,
    };
    mockedProcessRawChatHistory.mockReturnValue(new Map([[42, settledNode]]));
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ messages: [], packets: [] }),
    }) as unknown as typeof fetch;

    const stream = controlledResumeStream();
    mockedResumeStream.mockReturnValue(stream.generator());
    mockedProbe.mockResolvedValue({ processing: true, run_id: 42 });

    const attaching = attachToRunningRun(sessionId, 42);
    await tick();

    // Attached: streaming state restored, and the fake "terminated" answer is
    // gone from both the message text and the synthesized packet timeline.
    const session = useChatSessionStore.getState().sessions.get(sessionId);
    expect(session?.chatState).toBe("streaming");
    expect(node.message).toBe("");
    expect(node.packets).toHaveLength(0);
    expect(node.packetCount).toBe(0);

    // The run's first output arrives and takes over the timeline.
    stream.push(packet("agent_start"));
    await tick();
    expect(node.packets).toEqual([packet("agent_start")]);
    expect(node.packetCount).toBe(1);

    // Run completes: state settles from the persisted session.
    stream.finish();
    expect(await attaching).toBe(true);
    await tick();

    const settledSession = useChatSessionStore
      .getState()
      .sessions.get(sessionId);
    expect(settledSession?.chatState).toBe("input");
    expect(settledSession?.messageTree.get(42)?.message).toBe("Final answer");
  });

  it("refuses to attach twice to the same run", async () => {
    const sessionId = "session-resume-2";
    const node = placeholderNode(7);
    seedSession(sessionId, node);

    const stream = controlledResumeStream();
    mockedResumeStream.mockReturnValue(stream.generator());
    mockedProbe.mockResolvedValue({ processing: true, run_id: 7 });

    const first = attachToRunningRun(sessionId, 7);
    // Second caller for the same run must not start a competing tail.
    expect(await attachToRunningRun(sessionId, 7)).toBe(false);

    // Navigate away, then deliver a packet: the tail sees a stale session
    // and detaches instead of writing into the wrong session.
    useChatSessionStore.setState({ currentSessionId: "other-session" });
    stream.push(packet("agent_start"));
    expect(await first).toBe(false);
  });
});
