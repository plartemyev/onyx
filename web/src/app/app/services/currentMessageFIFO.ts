import {
  getChatSessionProcessingStatus,
  PacketType,
  resumeStream,
  SendMessageParams,
  sendMessage,
} from "./lib";
import { resilientPacketStream, StreamResyncMarker } from "./resilientStream";

/** What the FIFO hands to the stream consumer. */
type FifoItem = PacketType | StreamResyncMarker;

export class CurrentMessageFIFO {
  private stack: FifoItem[] = [];
  isComplete: boolean = false;
  error: string | null = null;
  /**
   * The run finished server-side while the client was disconnected; the
   * consumer must refetch the persisted session instead of trusting the
   * partially-delivered stream.
   */
  needsSessionRefresh: boolean = false;

  push(item: FifoItem) {
    this.stack.push(item);
  }

  nextPacket(): FifoItem | undefined {
    return this.stack.shift();
  }

  isEmpty(): boolean {
    return this.stack.length === 0;
  }
}

export interface UpdateCurrentMessageFIFOOptions {
  /** Fires once per drop streak, when the stream first needs a re-attach. */
  onReconnectStart?: () => void;
}

export async function updateCurrentMessageFIFO(
  stack: CurrentMessageFIFO,
  params: SendMessageParams,
  options?: UpdateCurrentMessageFIFOOptions
) {
  try {
    for await (const item of resilientPacketStream({
      sendConnect: async (signal) => sendMessage({ ...params, signal }),
      resumeConnect: async (signal) =>
        resumeStream(params.chatSessionId, 0, signal),
      probe: () => getChatSessionProcessingStatus(params.chatSessionId),
      signal: params.signal,
      onReconnectStart: options?.onReconnectStart,
      onSessionRefresh: () => {
        stack.needsSessionRefresh = true;
      },
    })) {
      if (params.signal?.aborted) {
        throw new Error("AbortError");
      }
      // Resync markers are forwarded: the consumer resets its accumulated
      // stream state and rebuilds it from the replay that follows.
      stack.push(item);
    }
  } catch (error: unknown) {
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        console.debug("Stream aborted");
      } else {
        stack.error = error.message;
      }
    } else {
      stack.error = String(error);
    }
  } finally {
    stack.isComplete = true;
  }
}
