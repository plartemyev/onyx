import { useEffect } from "react";
import { useTranslations } from "next-intl";
import { FiEye } from "react-icons/fi";
import {
  AnalyzeImageFinal,
  AnalyzeImagePacket,
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

interface AnalyzeImageState {
  files: { filename: string; file_id: string; annotation?: string | null }[];
  failures: string[];
  isAnalyzing: boolean;
  isComplete: boolean;
}

function constructAnalyzeImageState(
  packets: AnalyzeImagePacket[]
): AnalyzeImageState {
  const finalPacket = packets.find(
    (p) => p.obj.type === PacketType.ANALYZE_IMAGE_FINAL
  )?.obj as AnalyzeImageFinal | null;

  const hasStart = packets.some(
    (p) => p.obj.type === PacketType.ANALYZE_IMAGE_START
  );
  const hasEnd = packets.some(
    (p) =>
      p.obj.type === PacketType.SECTION_END || p.obj.type === PacketType.ERROR
  );

  return {
    files: finalPacket?.files ?? [],
    failures: finalPacket?.failures ?? [],
    isAnalyzing: hasStart && !finalPacket && !hasEnd,
    isComplete: hasStart && (!!finalPacket || hasEnd),
  };
}

export const AnalyzeImageRenderer: MessageRenderer<AnalyzeImagePacket, {}> = ({
  packets,
  onComplete,
  renderType,
  stopPacketSeen,
  children,
}) => {
  const t = useTranslations("chat.messages.timeline");
  const state = constructAnalyzeImageState(packets);

  useEffect(() => {
    if (state.isComplete) {
      onComplete();
    }
  }, [state.isComplete, onComplete]);

  const statusText = state.isComplete
    ? t("toolNames.analyzeImage")
    : t("header.analyzingImage.label");

  if (renderType === RenderType.COMPACT) {
    return children([
      {
        icon: FiEye,
        status: statusText,
        supportsCollapsible: true,
        timelineLayout: "timeline",
        content: <></>,
      },
    ]);
  }

  return children([
    {
      icon: FiEye,
      status: statusText,
      supportsCollapsible: true,
      timelineLayout: "timeline",
      content: (
        <Section gap={2} alignItems="start" height="fit">
          {state.isAnalyzing && !stopPacketSeen && <BlinkingBar />}
          {state.files.length > 0 && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {state.files.map((file) => (
                <div key={file.file_id} className="flex flex-col gap-1">
                  <InMessageImage
                    fileId={file.file_id}
                    fileName={file.filename}
                  />
                  {file.annotation && (
                    <Text as="span" mainUiMuted text03>
                      {file.annotation}
                    </Text>
                  )}
                </div>
              ))}
            </div>
          )}
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
