import { ChatFileType, FileDescriptor } from "@/app/app/interfaces";
import { isAudioFile, isVideoFile } from "@/lib/utils";

export function buildMediaUrl(fileId: string) {
  return `/api/chat/file/${fileId}`;
}

/**
 * Effective ChatFileType of a descriptor, falling back to the filename
 * extension for legacy descriptors: media files saved before AUDIO/VIDEO
 * existed were persisted as PLAIN_TEXT (e.g. agent-downloaded .wav files).
 */
export function resolveChatFileType(file: FileDescriptor): ChatFileType {
  if (file.type === ChatFileType.AUDIO || file.type === ChatFileType.VIDEO) {
    return file.type;
  }
  const name = file.name ?? "";
  if (isAudioFile(name)) return ChatFileType.AUDIO;
  if (isVideoFile(name)) return ChatFileType.VIDEO;
  return file.type;
}
