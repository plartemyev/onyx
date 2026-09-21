import { useEffect } from "react";
import { useTranslations } from "next-intl";
import { SvgDownload } from "@opal/icons";
import {
  DownloadToolFinal,
  DownloadToolPacket,
  PacketType,
} from "@/app/app/services/streamingModels";
import {
  MessageRenderer,
  RenderType,
} from "@/app/app/message/messageComponents/interfaces";
import { BlinkingBar } from "@/app/app/message/BlinkingBar";
import { Section } from "@/layouts/general-layouts";
import Text from "@/refresh-components/texts/Text";
import { InMessageImage } from "@/app/app/components/files/images/InMessageImage";
import {
  buildImgUrl,
  isImageFileName,
} from "@/app/app/components/files/images/utils";

interface DownloadToolState {
  files: { filename: string; file_id: string }[];
  failures: string[];
  isDownloading: boolean;
  isComplete: boolean;
}

function constructDownloadState(
  packets: DownloadToolPacket[]
): DownloadToolState {
  const finalPacket = packets.find(
    (p) => p.obj.type === PacketType.DOWNLOAD_TOOL_FINAL
  )?.obj as DownloadToolFinal | null;

  const hasStart = packets.some(
    (p) => p.obj.type === PacketType.DOWNLOAD_TOOL_START
  );
  const hasEnd = packets.some(
    (p) => p.obj.type === PacketType.SECTION_END || p.obj.type === PacketType.ERROR
  );

  return {
    files: finalPacket?.files ?? [],
    failures: finalPacket?.failures ?? [],
    isDownloading: hasStart && !finalPacket && !hasEnd,
    isComplete: hasStart && (!!finalPacket || hasEnd),
  };
}

export const DownloadToolRenderer: MessageRenderer<DownloadToolPacket, {}> = ({
  packets,
  onComplete,
  renderType,
  stopPacketSeen,
  children,
}) => {
  const t = useTranslations("chat.messages.timeline");
  const state = constructDownloadState(packets);

  useEffect(() => {
    if (state.isComplete) {
      onComplete();
    }
  }, [state.isComplete, onComplete]);

  const statusText = state.isComplete
    ? t("toolNames.downloadFile")
    : t("header.downloadingFile.label");

  if (renderType === RenderType.COMPACT) {
    return children([
      {
        icon: SvgDownload,
        status: statusText,
        supportsCollapsible: true,
        timelineLayout: "timeline",
        content: <></>,
      },
    ]);
  }

  const images = state.files.filter((file) => isImageFileName(file.filename));
  const otherFiles = state.files.filter(
    (file) => !isImageFileName(file.filename)
  );

  return children([
    {
      icon: SvgDownload,
      status: statusText,
      supportsCollapsible: true,
      timelineLayout: "timeline",
      content: (
        <Section gap={2} alignItems="start" height="fit">
          {state.isDownloading && !stopPacketSeen && <BlinkingBar />}
          {images.length > 0 && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {images.map((file) => (
                <InMessageImage
                  key={file.file_id}
                  fileId={file.file_id}
                  fileName={file.filename}
                />
              ))}
            </div>
          )}
          {otherFiles.map((file) => (
            <a
              key={file.file_id}
              href={buildImgUrl(file.file_id)}
              download={file.filename}
              className="text-sm text-text-03 underline hover:text-text-02 w-fit"
            >
              {file.filename}
            </a>
          ))}
          {state.failures.map((failure) => (
            <Text as="span" key={failure} mainUiMuted text04>
              {failure}
            </Text>
          ))}
        </Section>
      ),
    },
  ]);
};
