import { Section } from "@/layouts/general-layouts";
import { PreviewVariant } from "@/sections/modals/PreviewModal/interfaces";
import { DownloadButton } from "@/sections/modals/PreviewModal/variants/shared";
import { AUDIO_EXTENSIONS, VIDEO_EXTENSIONS } from "@/lib/utils";

const AUDIO_MIME_PREFIX = "audio/";
const VIDEO_MIME_PREFIX = "video/";

function extensionPattern(extensions: readonly string[]): RegExp {
  return new RegExp(`\\.(${extensions.join("|")})$`, "i");
}

function mediaMatcher(prefix: string, extensions: readonly string[]) {
  const pattern = extensionPattern(extensions);
  return (name: string | null, mime: string): boolean => {
    const normalized = mime.split(";")[0]?.trim().toLowerCase() ?? "";
    return normalized.startsWith(prefix) || pattern.test(name ?? "");
  };
}

/**
 * Media variants stream from the raw chat-file URL instead of materializing
 * the blob: attachments can be tens of MB, and <audio>/<video> with
 * `preload="metadata"` only pull what they play.
 */
function renderMediaPlayer(
  kind: "audio" | "video",
  ctx: Parameters<PreviewVariant["renderContent"]>[0]
) {
  return (
    <Section className="flex-1 min-h-0 p-4">
      {kind === "audio" ? (
        <audio
          controls
          preload="metadata"
          src={ctx.fileUrl}
          className="w-full max-w-md"
        >
          <track kind="captions" />
        </audio>
      ) : (
        <video
          controls
          preload="metadata"
          src={ctx.fileUrl}
          className="max-h-full max-w-full rounded-lg"
        >
          <track kind="captions" />
        </video>
      )}
    </Section>
  );
}

const audioMatches = mediaMatcher(AUDIO_MIME_PREFIX, AUDIO_EXTENSIONS);
const videoMatches = mediaMatcher(VIDEO_MIME_PREFIX, VIDEO_EXTENSIONS);

export const audioVariant: PreviewVariant = {
  matches: audioMatches,
  width: "md",
  height: "fit",
  needsTextContent: false,
  needsRawStream: true,
  codeBackground: false,
  headerDescription: () => "",

  renderContent: (ctx) => renderMediaPlayer("audio", ctx),

  renderFooterLeft: () => null,

  renderFooterRight: (ctx) => (
    <Section flexDirection="row" width="fit">
      <DownloadButton fileUrl={ctx.fileUrl} fileName={ctx.fileName} />
    </Section>
  ),
};

export const videoVariant: PreviewVariant = {
  matches: videoMatches,
  width: "lg",
  height: "full",
  needsTextContent: false,
  needsRawStream: true,
  codeBackground: false,
  headerDescription: () => "",

  renderContent: (ctx) => renderMediaPlayer("video", ctx),

  renderFooterLeft: () => null,

  renderFooterRight: (ctx) => (
    <Section flexDirection="row" width="fit">
      <DownloadButton fileUrl={ctx.fileUrl} fileName={ctx.fileName} />
    </Section>
  ),
};
