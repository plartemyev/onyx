import { useTranslations } from "next-intl";
import { FetchToolPacket } from "@/app/app/services/streamingModels";
import {
  MessageRenderer,
  RenderType,
} from "@/app/app/message/messageComponents/interfaces";
import { BlinkingBar } from "@/app/app/message/BlinkingBar";
import { OnyxDocument } from "@/lib/search/interfaces";
import { openExternalLink } from "@/lib/search/utils";
import { ValidSources } from "@/lib/types";
import { SearchChipList, SourceInfo } from "../search/SearchChipList";
import { documentToSourceInfo } from "../search/searchStateUtils";
import {
  constructCurrentFetchState,
  INITIAL_URLS_TO_SHOW,
  URLS_PER_EXPANSION,
} from "./fetchStateUtils";
import {
  formatDataSize,
  formatKib,
  formatTokenCount,
  StepHeaderSummary,
} from "../StepHeaderSummary";
import Text from "@/refresh-components/texts/Text";
import { SvgCircle } from "@opal/icons";

const urlToSourceInfo = (url: string, index: number): SourceInfo => ({
  id: `url-${index}`,
  title: url,
  sourceType: ValidSources.Web,
  sourceUrl: url,
});

/**
 * FetchToolRenderer - Renders URL fetch/open tool execution steps
 *
 * The step header carries the measured size/token cost of the LLM-facing
 * response once the documents packet arrives; the document chips are collapsed
 * by default and revealed by expanding the step.
 *
 * RenderType modes:
 * - FULL: Shows all details (URLs being opened + reading). Header passed as `status` prop.
 *         Used when step is expanded in timeline.
 * - COMPACT: Single summary line with the measured stats. Header passed as `status` prop.
 *            Used when step is collapsed in timeline, still wrapped in StepContainer.
 * - HIGHLIGHT: Shows URL list with header embedded directly in content.
 *              No StepContainer wrapper. Used for parallel streaming preview.
 */
export const FetchToolRenderer: MessageRenderer<FetchToolPacket, {}> = ({
  packets,
  onComplete,
  animate,
  stopPacketSeen,
  renderType,
  children,
}) => {
  const t = useTranslations("chat.messages.timeline");
  const fetchState = constructCurrentFetchState(packets);
  const {
    urls,
    documents,
    hasStarted,
    isLoading,
    isComplete,
    contentBytes,
    tokenCount,
    docStats,
  } = fetchState;
  const isCompact = renderType === RenderType.COMPACT;
  const isHighlight = renderType === RenderType.HIGHLIGHT;

  // Small per-URL stat suffix for each document chip ("45.2 KiB · 11.3k tokens")
  const getDocumentStats = (doc: OnyxDocument): string | undefined => {
    const stat = docStats[doc.document_id];
    if (!stat) return undefined;
    const parts: string[] = [];
    if (stat.content_bytes != null) {
      parts.push(formatDataSize(stat.content_bytes));
    }
    if (stat.token_count != null) {
      parts.push(
        t("fetch.summary.tokens", {
          tokens: formatTokenCount(stat.token_count),
        })
      );
    }
    return parts.length > 0 ? parts.join(" · ") : undefined;
  };

  const headerLabel = t("fetch.reading.status");
  const headerStats: string[] = [];
  if (contentBytes != null) {
    headerStats.push(
      t("fetch.summary.size", { size: formatKib(contentBytes) })
    );
  }
  if (tokenCount != null) {
    headerStats.push(
      t("fetch.summary.tokens", { tokens: formatTokenCount(tokenCount) })
    );
  }
  const header = <StepHeaderSummary label={headerLabel} stats={headerStats} />;

  if (!hasStarted) {
    return children([
      {
        icon: SvgCircle,
        status: headerLabel,
        content: <div />,
        supportsCollapsible: false,
        timelineLayout: "timeline",
      },
    ]);
  }

  const displayDocuments = documents.length > 0;
  const displayUrls = !displayDocuments && isComplete && urls.length > 0;

  const chips =
    displayDocuments || displayUrls ? (
      displayDocuments ? (
        <SearchChipList
          items={documents}
          initialCount={INITIAL_URLS_TO_SHOW}
          expansionCount={URLS_PER_EXPANSION}
          getKey={(doc: OnyxDocument) => doc.document_id}
          toSourceInfo={(doc: OnyxDocument) => documentToSourceInfo(doc)}
          onClick={(doc: OnyxDocument) => {
            if (doc.link) openExternalLink(doc.link);
          }}
          emptyState={!stopPacketSeen ? <BlinkingBar /> : undefined}
          getItemStats={getDocumentStats}
        />
      ) : (
        <SearchChipList
          items={urls}
          initialCount={INITIAL_URLS_TO_SHOW}
          expansionCount={URLS_PER_EXPANSION}
          getKey={(url: string) => url}
          toSourceInfo={urlToSourceInfo}
          onClick={(url: string) => openExternalLink(url)}
          emptyState={!stopPacketSeen ? <BlinkingBar /> : undefined}
        />
      )
    ) : (
      <div className="flex flex-wrap gap-x-2 gap-y-2 ms-1">
        {!stopPacketSeen && <BlinkingBar />}
      </div>
    );

  // COMPACT mode: single summary line once documents/URLs exist. While still
  // loading, fall through to the chips below — the collapsed streaming
  // container renders only `content`, so an empty summary line would hide the
  // reading indicator.
  if (isCompact && (displayDocuments || displayUrls)) {
    return children([
      {
        icon: SvgCircle,
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
            <Text as="p" text02 className="text-sm mb-1">
              {header}
            </Text>
            {chips}
          </div>
        ),
      },
    ]);
  }

  return children([
    {
      icon: SvgCircle,
      status: header,
      supportsCollapsible: true,
      alwaysCollapsible: true,
      timelineLayout: "timeline",
      content: <div className="flex flex-col">{chips}</div>,
    },
  ]);
};
