import React from "react";
import { useTranslations } from "next-intl";
import { SvgSearch, SvgGlobe } from "@opal/icons";
import { SearchToolPacket } from "@/app/app/services/streamingModels";
import {
  MessageRenderer,
  RenderType,
} from "@/app/app/message/messageComponents/interfaces";
import { BlinkingBar } from "@/app/app/message/BlinkingBar";
import { openExternalLink } from "@/lib/search/utils";
import { ValidSources } from "@/lib/types";
import { SearchChipList, SourceInfo } from "./SearchChipList";
import {
  constructCurrentSearchState,
  documentToSourceInfo,
  INITIAL_QUERIES_TO_SHOW,
  QUERIES_PER_EXPANSION,
  INITIAL_RESULTS_TO_SHOW,
  RESULTS_PER_EXPANSION,
} from "./searchStateUtils";
import { StepHeaderSummary, StatsRow } from "../StepHeaderSummary";
import Text from "@/refresh-components/texts/Text";

const queryToSourceInfo = (query: string, index: number): SourceInfo => ({
  id: `query-${index}`,
  title: query,
  sourceType: ValidSources.Web,
  icon: SvgSearch,
});

/**
 * WebSearchToolRenderer - Renders web search tool execution steps
 *
 * Shows the queries and the result documents returned by the search. Both are
 * collapsed by default: the step header carries the query/result counts and
 * expanding the step reveals the chips.
 *
 * RenderType modes:
 * - FULL: Shows query + result chips. Used when step is expanded in timeline.
 * - COMPACT: Single summary line with counts. Used when step is collapsed in
 *            timeline (the default for completed steps).
 * - HIGHLIGHT: Shows chips with header embedded directly in content.
 *              No StepContainer wrapper. Used for parallel streaming preview.
 * - INLINE: Shows chips + counts for collapsed streaming view.
 */
export const WebSearchToolRenderer: MessageRenderer<SearchToolPacket, {}> = ({
  packets,
  onComplete,
  animate,
  stopPacketSeen,
  renderType,
  children,
}) => {
  const t = useTranslations("chat.messages.timeline");
  const searchState = constructCurrentSearchState(packets);
  const { queries, results } = searchState;

  const isHighlight = renderType === RenderType.HIGHLIGHT;
  const isInline = renderType === RenderType.INLINE;
  const isCompact = renderType === RenderType.COMPACT;

  const headerLabel = t("webSearch.searching.status");
  const headerStats: string[] = [];
  if (queries.length > 0) {
    headerStats.push(t("webSearch.summary.queries", { count: queries.length }));
  }
  if (results.length > 0) {
    headerStats.push(t("webSearch.summary.results", { count: results.length }));
  }
  const header = <StepHeaderSummary label={headerLabel} stats={headerStats} />;

  if (queries.length === 0) {
    return children([
      {
        icon: SvgGlobe,
        status: headerLabel,
        content: <div />,
        supportsCollapsible: false,
        timelineLayout: "timeline",
      },
    ]);
  }

  const queriesChips = (
    <SearchChipList
      items={queries}
      initialCount={INITIAL_QUERIES_TO_SHOW}
      expansionCount={QUERIES_PER_EXPANSION}
      getKey={(_, index) => index}
      toSourceInfo={queryToSourceInfo}
      emptyState={!stopPacketSeen ? <BlinkingBar /> : undefined}
      showDetailsCard={false}
      isQuery={true}
    />
  );

  const resultsChips =
    results.length > 0 ? (
      <SearchChipList
        items={results}
        initialCount={INITIAL_RESULTS_TO_SHOW}
        expansionCount={RESULTS_PER_EXPANSION}
        getKey={(doc) => doc.document_id}
        toSourceInfo={(doc) => documentToSourceInfo(doc)}
        onClick={(doc) => {
          if (doc.link) openExternalLink(doc.link);
        }}
        emptyState={!stopPacketSeen ? <BlinkingBar /> : undefined}
      />
    ) : null;

  // COMPACT mode: single summary line; expanding the step reveals the chips
  if (isCompact) {
    return children([
      {
        icon: SvgGlobe,
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
            {queriesChips}
            {resultsChips}
          </div>
        ),
      },
    ]);
  }

  // INLINE mode: chips + counts for collapsed streaming view. The collapsed
  // streaming container renders only `content`, so the stats ride inside it.
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
            {queriesChips}
            {resultsChips}
          </div>
        ),
      },
    ]);
  }

  // FULL mode: query + result chips under the summary header
  return children([
    {
      icon: SvgGlobe,
      status: header,
      content: (
        <div className="flex flex-col">
          {queriesChips}
          {resultsChips}
        </div>
      ),
      supportsCollapsible: true,
      alwaysCollapsible: true,
      timelineLayout: "timeline",
    },
  ]);
};
