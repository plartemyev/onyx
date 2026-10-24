"use client";

import { Button, Text } from "@opal/components";
import { SvgAudioFile, SvgDownload } from "@opal/icons";
import { Hoverable } from "@opal/core";
import { useTranslations } from "next-intl";
import { buildMediaUrl } from "@/app/app/components/files/media/utils";

export type MediaKind = "audio" | "video";

interface InMessageMediaProps {
  fileId: string;
  fileName?: string;
  kind: MediaKind;
}

function MediaDownloadButton({
  fileUrl,
  fileName,
  tooltip,
}: {
  fileUrl: string;
  fileName: string;
  tooltip: string;
}) {
  return (
    <a href={fileUrl} download={fileName} className="shrink-0">
      <Button
        prominence="tertiary"
        size="sm"
        icon={SvgDownload}
        tooltip={tooltip}
      />
    </a>
  );
}

/**
 * Inline player for audio/video attachments in a message. Streams from the
 * raw chat-file URL (metadata preload only — attachments can be tens of MB);
 * the download button is a plain same-origin anchor so the browser streams
 * the save without materializing the bytes in JS.
 */
export default function InMessageMedia({
  fileId,
  fileName,
  kind,
}: InMessageMediaProps) {
  const t = useTranslations("chat.files.inMessageMedia");

  const fileUrl = buildMediaUrl(fileId);
  const downloadName = fileName || fileId;

  if (kind === "video") {
    return (
      <Hoverable.Root group={`media-${fileId}`} width="fit">
        <div className="relative max-w-md">
          <video
            controls
            preload="metadata"
            src={fileUrl}
            aria-label={t("videoPlayer.label", { name: downloadName })}
            className="w-full rounded-lg border border-border-01 bg-background-tint-02"
          >
            <track kind="captions" />
          </video>
          <div className="absolute bottom-2 end-2 z-10">
            <Hoverable.Item group={`media-${fileId}`} variant="appear-on-hover">
              <MediaDownloadButton
                fileUrl={fileUrl}
                fileName={downloadName}
                tooltip={t("downloadButton.tooltip")}
              />
            </Hoverable.Item>
          </div>
        </div>
      </Hoverable.Root>
    );
  }

  return (
    <div className="w-full max-w-md rounded-12 border border-border-01 bg-background-tint-01 p-2">
      <div className="flex items-center gap-2 px-1 pb-1">
        <SvgAudioFile size={16} className="shrink-0 text-text-03" />
        <div className="min-w-0 flex-1">
          <Text
            font="main-ui-body"
            color="text-03"
            maxLines={1}
            wordWrap="wrap-anywhere"
          >
            {downloadName}
          </Text>
        </div>
        <MediaDownloadButton
          fileUrl={fileUrl}
          fileName={downloadName}
          tooltip={t("downloadButton.tooltip")}
        />
      </div>
      <audio
        controls
        preload="metadata"
        src={fileUrl}
        aria-label={t("audioPlayer.label", { name: downloadName })}
        className="w-full"
      >
        <track kind="captions" />
      </audio>
    </div>
  );
}
