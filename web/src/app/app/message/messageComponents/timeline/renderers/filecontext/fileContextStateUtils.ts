import {
  PacketType,
  ChatFileContext,
  FileContextPacket,
} from "@/app/app/services/streamingModels";

export interface FileContextState {
  /** Documents attached to the message, in upload order */
  files: ChatFileContext[];
  isComplete: boolean;
}

/** Constructs the current file-context state from the grouped packets. */
export const constructFileContextState = (
  packets: FileContextPacket[]
): FileContextState => {
  const seenIds = new Set<string>();
  const files: ChatFileContext[] = [];
  for (const packet of packets) {
    if (packet.obj.type !== PacketType.FILE_CONTEXT) continue;
    if (seenIds.has(packet.obj.file_id)) continue;
    seenIds.add(packet.obj.file_id);
    files.push(packet.obj);
  }
  const isComplete = packets.some(
    (packet) =>
      packet.obj.type === PacketType.SECTION_END ||
      packet.obj.type === PacketType.ERROR
  );
  return { files, isComplete };
};
