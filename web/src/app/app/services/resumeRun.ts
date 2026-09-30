import { Packet } from "@/app/app/services/streamingModels";
import {
  getChatSessionProcessingStatus,
  processRawChatHistory,
  resumeStream,
} from "@/app/app/services/lib";
import {
  isStreamResyncMarker,
  resilientPacketStream,
} from "@/app/app/services/resilientStream";
import { BackendChatSession, Message } from "@/app/app/interfaces";
import { useChatSessionStore } from "@/app/app/stores/useChatSessionStore";

// Runs currently being re-attached; module-level so concurrent callers
// (page-load resume, submit-guard re-attach) can't start a second tail for
// the same run.
const resumingRuns = new Set<number>();

async function fetchSessionMessages(
  sessionId: string
): Promise<Map<number, Message> | null> {
  try {
    const response = await fetch(`/api/chat/get-chat-session/${sessionId}`);
    if (!response.ok) {
      return null;
    }
    const session: BackendChatSession = await response.json();
    return processRawChatHistory(session.messages, session.packets);
  } catch {
    return null;
  }
}

/**
 * Re-attach this client to a run that is still processing server-side:
 * replay its buffered stream into the live timeline and tail it until done,
 * surviving further transport drops. Converges to the persisted session
 * state when the tail ends for any reason. Returns false when the run has no
 * live timeline here (e.g. multi-model runs, whose run id is the user
 * message) or another tail is already attached.
 */
export async function attachToRunningRun(
  sessionId: string,
  runId: number
): Promise<boolean> {
  if (resumingRuns.has(runId)) {
    return false;
  }
  const store = useChatSessionStore.getState();

  let messageMap = store.sessions.get(sessionId)?.messageTree ?? null;
  let node = messageMap?.get(runId);
  if (!node || node.type !== "assistant") {
    // The local tree predates the run's placeholder (stale tab, submit-guard
    // after an error cleanup): rebuild from the backend before attaching.
    const rebuilt = await fetchSessionMessages(sessionId);
    if (!rebuilt) {
      return false;
    }
    useChatSessionStore
      .getState()
      .updateSessionAndMessageTree(sessionId, rebuilt);
    messageMap = rebuilt;
    node = rebuilt.get(runId);
    if (!node || node.type !== "assistant") {
      return false;
    }
  }

  // Added and deleted in this function only, so an entry can never outlive
  // its tail.
  resumingRuns.add(runId);
  // A run is in flight: restore the streaming state so the input bar shows
  // the stop button (which fences the session server-side) and enqueues
  // follow-ups instead of starting a second concurrent run.
  useChatSessionStore.getState().updateChatState(sessionId, "streaming");
  // The reserved row's placeholder text would render above the live timeline.
  node.message = "";
  const accumulated: Packet[] = [];
  let lastFlush = 0;
  let trailingFlush: ReturnType<typeof setTimeout> | null = null;
  // updateSessionAndMessageTree re-points currentSessionId at this session;
  // once the user navigates elsewhere, any further store write from this
  // tail would hijack their new session's sends.
  const stillCurrent = () =>
    useChatSessionStore.getState().currentSessionId === sessionId;
  const flush = () => {
    if (!stillCurrent() || !messageMap) {
      return;
    }
    node!.packets = [...accumulated];
    // AgentMessage's memo compares packetCount, not the packets array.
    node!.packetCount = accumulated.length;
    useChatSessionStore
      .getState()
      .updateSessionAndMessageTree(sessionId, new Map(messageMap));
  };

  try {
    for await (const rawItem of resilientPacketStream({
      resumeConnect: async (signal) => resumeStream(sessionId, 0, signal),
      probe: () => getChatSessionProcessingStatus(sessionId),
    })) {
      if (!stillCurrent()) {
        return false;
      }
      // Re-assert on every received packet: navigating away and back
      // re-initializes the session store with chatState "input" while this
      // tail is live.
      useChatSessionStore.getState().updateChatState(sessionId, "streaming");
      if (isStreamResyncMarker(rawItem)) {
        accumulated.length = 0;
        continue;
      }
      if (!Object.hasOwn(rawItem, "obj")) {
        continue;
      }
      // SAFETY: resync markers are excluded above; every remaining FIFO item
      // with an "obj" key is a wire Packet (see streamingModels.ts).
      const packet = rawItem as Packet;
      accumulated.push(packet);
      const now = Date.now();
      if (now - lastFlush >= 100) {
        lastFlush = now;
        flush();
      } else if (trailingFlush === null) {
        // A burst's last packets would otherwise wait for the NEXT packet to
        // render — during quiet phases that's minutes.
        trailingFlush = setTimeout(() => {
          trailingFlush = null;
          lastFlush = Date.now();
          flush();
        }, 120);
      }
    }
  } catch (error) {
    console.error("Failed to tail in-flight run", { runId, error });
  } finally {
    if (trailingFlush !== null) {
      clearTimeout(trailingFlush);
    }
    resumingRuns.delete(runId);
    if (stillCurrent()) {
      // The tail ended: release the streaming state so the input bar stops
      // offering stop and accepts normal sends again.
      useChatSessionStore.getState().updateChatState(sessionId, "input");
      flush();
      // Settle final state (message text, citations, documents) from the
      // persisted session — also the recovery path when the buffer replay
      // ended early (gap) or the run finished while disconnected.
      const settled = await fetchSessionMessages(sessionId);
      if (settled && stillCurrent()) {
        useChatSessionStore
          .getState()
          .updateSessionAndMessageTree(sessionId, settled);
      }
    }
  }
  return true;
}
