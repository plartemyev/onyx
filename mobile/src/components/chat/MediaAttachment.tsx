// Playable audio/video message attachment. Audio streams inline through
// expo-audio; a video card opens a full-screen expo-video player. Download
// fetches the bytes with the bearer token into the cache, then hands the file
// to the OS share sheet (Save to Files / any player app).
import { memo, useCallback, useMemo, useState } from "react";
import { Modal, Pressable, View } from "react-native";
import { useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { useVideoPlayer, VideoView } from "expo-video";

import { getBaseUrl } from "@/api/config";
import { getValidToken } from "@/api/auth/refreshState";
import { Icon } from "@/components/ui/icon";
import { Spinner } from "@/components/ui/spinner";
import { Text } from "@/components/ui/text";
import { toast } from "@/hooks/useToast";
import { useAuthToken } from "@/hooks/useAuthToken";
import SvgDownload from "@/icons/download";
import SvgMusicSmall from "@/icons/music-small";
import SvgPauseCircle from "@/icons/pause-circle";
import SvgPlayCircle from "@/icons/play-circle";
import SvgVideoFile from "@/icons/video-file";
import { extensionOf } from "@/lib/files";

export type MediaKind = "audio" | "video";

interface MediaAttachmentProps {
  fileId: string;
  fileName: string;
  kind: MediaKind;
}

// Media element MIME for the share sheet, guessed from the extension. The
// server knows the true type; this only has to be good enough for the OS to
// pick a player.
const MIME_BY_EXTENSION: Record<string, string> = {
  MP3: "audio/mpeg",
  WAV: "audio/x-wav",
  OGG: "audio/ogg",
  OGA: "audio/ogg",
  M4A: "audio/mp4",
  AAC: "audio/aac",
  FLAC: "audio/flac",
  OPUS: "audio/opus",
  WEBA: "audio/webm",
  MP4: "video/mp4",
  WEBM: "video/webm",
  MOV: "video/quicktime",
  MKV: "video/x-matroska",
  AVI: "video/x-msvideo",
  MPG: "video/mpeg",
  MPEG: "video/mpeg",
};

function shareMimeType(fileName: string, kind: MediaKind): string {
  return MIME_BY_EXTENSION[extensionOf(fileName)] ?? `${kind}/*`;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const wholeSeconds = Math.floor(seconds % 60);
  return `${minutes}:${String(wholeSeconds).padStart(2, "0")}`;
}

// The cache filename also becomes the share-sheet title, so keep it a bare
// basename even if a descriptor name carried a path.
function cacheFileName(fileName: string): string {
  const basename = fileName.split("/").pop()?.trim();
  return basename ? basename : "attachment";
}

function useMediaDownload(
  fileId: string,
  fileName: string,
  kind: MediaKind,
): { download: () => void; downloading: boolean } {
  const [downloading, setDownloading] = useState(false);

  const download = useCallback(async () => {
    setDownloading(true);
    try {
      const token = await getValidToken();
      const name = cacheFileName(fileName);
      const file = await File.downloadFileAsync(
        `${getBaseUrl()}/chat/file/${encodeURIComponent(fileId)}`,
        new File(Paths.cache, name),
        {
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
          idempotent: true,
        },
      );
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(file.uri, {
          mimeType: shareMimeType(fileName, kind),
          dialogTitle: name,
        });
      } else {
        toast.info(`Saved to cache as ${name}`);
      }
    } catch {
      toast.error("Couldn't download this file.");
    } finally {
      setDownloading(false);
    }
  }, [fileId, fileName, kind]);

  return { download: () => void download(), downloading };
}

function DownloadButton({
  fileId,
  fileName,
  kind,
}: {
  fileId: string;
  fileName: string;
  kind: MediaKind;
}) {
  const { download, downloading } = useMediaDownload(fileId, fileName, kind);
  return (
    <Pressable
      onPress={download}
      hitSlop={8}
      disabled={downloading}
      accessibilityRole="button"
      accessibilityLabel={`Download ${fileName}`}
    >
      {downloading ? (
        <Spinner size={16} />
      ) : (
        <Icon as={SvgDownload} size={16} className="text-text-03" />
      )}
    </Pressable>
  );
}

// Auth'd streaming source for the native players; null until the bearer
// resolves (mirrors BearerImage — the URL isn't auth-keyed, so a shared cache
// would leak one account's media to the next after an instance switch).
function useMediaSource(
  fileId: string,
): { uri: string; headers: Record<string, string> } | null {
  const token = useAuthToken();
  return useMemo(() => {
    if (!token) return null;
    return {
      uri: `${getBaseUrl()}/chat/file/${encodeURIComponent(fileId)}`,
      headers: { Authorization: `Bearer ${token}` },
    };
  }, [fileId, token]);
}

function AudioAttachment({
  fileId,
  fileName,
}: {
  fileId: string;
  fileName: string;
}) {
  const source = useMediaSource(fileId);
  const player = useAudioPlayer(source);
  const status = useAudioPlayerStatus(player);

  const duration = Number.isFinite(status.duration) ? status.duration : 0;
  const progress =
    duration > 0 ? Math.min(status.currentTime / duration, 1) : 0;
  const timeLabel = `${formatTime(status.currentTime)} / ${formatTime(duration)}`;

  return (
    <View
      testID="media-audio-card"
      className="w-full max-w-[320px] gap-8 rounded-12 border border-border-01 bg-background-tint-01 px-12 py-8"
    >
      <View className="flex-row items-center gap-8">
        <Pressable
          onPress={() => (status.playing ? player.pause() : player.play())}
          disabled={!source}
          hitSlop={8}
          accessibilityRole="button"
          accessibilityLabel={
            status.playing ? `Pause ${fileName}` : `Play ${fileName}`
          }
        >
          <Icon
            as={status.playing ? SvgPauseCircle : SvgPlayCircle}
            size={24}
            className="text-text-02"
          />
        </Pressable>
        <View className="min-w-0 flex-1 gap-2">
          <Text font="main-ui-body" color="text-04" numberOfLines={1}>
            {fileName}
          </Text>
          <Text font="secondary-body" color="text-03">
            {source ? timeLabel : "Sign in to play"}
          </Text>
        </View>
        <Icon as={SvgMusicSmall} size={16} className="text-text-03" />
        <DownloadButton fileId={fileId} fileName={fileName} kind="audio" />
      </View>
      <View className="h-2 rounded-04 bg-background-tint-02">
        <View
          className="h-2 rounded-04 bg-action-selection-05"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
      </View>
    </View>
  );
}

function VideoAttachment({
  fileId,
  fileName,
}: {
  fileId: string;
  fileName: string;
}) {
  const source = useMediaSource(fileId);
  const [playerOpen, setPlayerOpen] = useState(false);
  const player = useVideoPlayer(playerOpen ? source : null);

  return (
    <>
      <Modal
        visible={playerOpen}
        animationType="slide"
        onRequestClose={() => setPlayerOpen(false)}
      >
        <View className="flex-1 bg-background-neutral-inverted-01">
          <VideoView
            player={player}
            contentFit="contain"
            fullscreenOptions={{ enable: true }}
            style={{ flex: 1 }}
          />
          <View className="flex-row items-center justify-between px-16 py-12">
            <Text
              font="main-ui-body"
              color="text-inverted-04"
              numberOfLines={1}
            >
              {fileName}
            </Text>
            <Pressable
              onPress={() => setPlayerOpen(false)}
              hitSlop={8}
              accessibilityRole="button"
              accessibilityLabel="Close video player"
            >
              <Text font="main-ui-action" color="text-inverted-03">
                Done
              </Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      <Pressable
        testID="media-video-card"
        onPress={() => setPlayerOpen(true)}
        disabled={!source}
        className="w-full max-w-[320px] flex-row items-center gap-8 rounded-12 border border-border-01 bg-background-tint-01 px-12 py-8"
        accessibilityRole="button"
        accessibilityLabel={`Play video ${fileName}`}
      >
        <Icon as={SvgPlayCircle} size={24} className="text-text-02" />
        <View className="min-w-0 flex-1 gap-2">
          <Text font="main-ui-body" color="text-04" numberOfLines={1}>
            {fileName}
          </Text>
          <Text font="secondary-body" color="text-03">
            {source ? "Tap to play" : "Sign in to play"}
          </Text>
        </View>
        <Icon as={SvgVideoFile} size={16} className="text-text-03" />
        <DownloadButton fileId={fileId} fileName={fileName} kind="video" />
      </Pressable>
    </>
  );
}

export const MediaAttachment = memo(function MediaAttachment({
  fileId,
  fileName,
  kind,
}: MediaAttachmentProps) {
  if (kind === "video") {
    return <VideoAttachment fileId={fileId} fileName={fileName} />;
  }
  return <AudioAttachment fileId={fileId} fileName={fileName} />;
});
