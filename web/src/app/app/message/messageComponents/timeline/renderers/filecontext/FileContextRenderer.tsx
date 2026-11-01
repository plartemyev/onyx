import React from "react";
import { useTranslations } from "next-intl";
import { SvgFileText } from "@opal/icons";
import { FileContextPacket } from "@/app/app/services/streamingModels";
import {
  MessageRenderer,
  RenderType,
} from "@/app/app/message/messageComponents/interfaces";
import { constructFileContextState } from "./fileContextStateUtils";
import {
  formatDataSize,
  formatTokenCount,
  StepHeaderSummary,
  StatsRow,
} from "../StepHeaderSummary";
import Text from "@/refresh-components/texts/Text";

/**
 * FileContextRenderer - Renders the documents uploaded with the user's
 * message that were read into the LLM context.
 *
 * Collapsed by default: the step header carries the aggregate numbers
 * (document count, size, token cost) and expanding it lists each document
 * with its own numbers.
 *
 * RenderType modes:
 * - FULL: Shows the per-document rows. Used when step is expanded in timeline.
 * - COMPACT: Single summary line with aggregate stats. Used when step is
 *            collapsed in timeline (the default for completed steps).
 * - HIGHLIGHT: Shows rows with header embedded directly in content.
 *              No StepContainer wrapper. Used for parallel streaming preview.
 * - INLINE: Shows rows + aggregate stats for collapsed streaming view.
 */
export const FileContextRenderer: MessageRenderer<FileContextPacket, {}> = ({
  packets,
  onComplete,
  animate,
  stopPacketSeen,
  renderType,
  children,
}) => {
  const t = useTranslations("chat.messages.timeline");
  const { files, isComplete } = constructFileContextState(packets);

  const isHighlight = renderType === RenderType.HIGHLIGHT;
  const isInline = renderType === RenderType.INLINE;
  const isCompact = renderType === RenderType.COMPACT;

  const headerLabel = t("fileContext.reading.status");

  const totalBytes = files.reduce(
    (sum, file) => sum + (file.content_bytes ?? 0),
    0
  );
  const totalTokens = files.reduce(
    (sum, file) => sum + (file.token_count ?? 0),
    0
  );
  const headerStats: string[] = [
    t("fileContext.summary.files", { count: files.length }),
  ];
  if (totalBytes > 0) headerStats.push(formatDataSize(totalBytes));
  if (totalTokens > 0) {
    headerStats.push(
      t("fileContext.summary.tokens", {
        tokens: formatTokenCount(totalTokens),
      })
    );
  }
  const header = <StepHeaderSummary label={headerLabel} stats={headerStats} />;

  // Per-document rows: name + its own size/token numbers when measured
  const fileRows = files.map((file) => {
    const stats: string[] = [];
    if (file.content_bytes != null) {
      stats.push(formatDataSize(file.content_bytes));
    }
    if (file.token_count != null) {
      stats.push(
        t("fileContext.summary.tokens", {
          tokens: formatTokenCount(file.token_count),
        })
      );
    }
    return (
      <Text key={file.file_id} as="p" text03 mainUiMuted className="min-w-0">
        <StepHeaderSummary label={file.file_name} stats={stats} />
      </Text>
    );
  });

  // COMPACT mode: single summary line; expanding the step reveals the rows.
  // While the step is still open (right after send, before the next step
  // starts), keep the rows visible — the collapsed streaming container
  // renders only `content`, so a summary-only line would hide them.
  if (isCompact && (isComplete || stopPacketSeen)) {
    return children([
      {
        icon: SvgFileText,
        status: header,
        content: <div />,
        supportsCollapsible: true,
        alwaysCollapsible: true,
        timelineLayout: "timeline",
      },
    ]);
  }

  // HIGHLIGHT mode: header embedded in content, no StepContainer
  if (isHighlight) {
    return children([
      {
        icon: null,
        status: null,
        supportsCollapsible: false,
        timelineLayout: "content",
        content: (
          <div className="flex flex-col">
            <Text as="p" text04 mainUiMuted className="mb-1">
              {header}
            </Text>
            {fileRows}
          </div>
        ),
      },
    ]);
  }

  // INLINE mode: rows + aggregate stats for collapsed streaming view. The
  // collapsed streaming container renders only `content`, so the stats ride
  // inside it.
  if (isInline) {
    return children([
      {
        icon: null,
        status: header,
        supportsCollapsible: false,
        timelineLayout: "content",
        content: (
          <div className="flex flex-col">
            {headerStats.length > 0 && (
              <Text as="p" text04 mainUiMuted className="mb-1">
                <StatsRow stats={headerStats} />
              </Text>
            )}
            {fileRows}
          </div>
        ),
      },
    ]);
  }

  // FULL mode: per-document rows under the summary header
  return children([
    {
      icon: SvgFileText,
      status: header,
      content: <div className="flex flex-col">{fileRows}</div>,
      supportsCollapsible: true,
      alwaysCollapsible: true,
      timelineLayout: "timeline",
    },
  ]);
};
