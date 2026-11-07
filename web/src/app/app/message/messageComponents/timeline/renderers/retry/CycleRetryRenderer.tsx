"use client";

import { useTranslations } from "next-intl";
import { CycleRetryNoticePacket } from "@/app/app/services/streamingModels";
import { MessageRenderer } from "@/app/app/message/messageComponents/interfaces";
import Text from "@/refresh-components/texts/Text";
import { SvgRefreshCw } from "@opal/icons";

/**
 * CycleRetryRenderer — one-line timeline notice for automatic turn recovery.
 *
 * Emitted live when the chat loop retries a failed cycle (transport drop,
 * stream-budget hit, empty final answer). Ephemeral by design: the packet has
 * no DB replay counterpart, so it only appears while the turn streams.
 */
export const CycleRetryRenderer: MessageRenderer<
  CycleRetryNoticePacket,
  {}
> = ({ packets, children }) => {
  const t = useTranslations("chat.messages.timeline");

  const notice = packets[0];
  const reasonLabel = (() => {
    switch (notice?.obj.reason) {
      case "transport_error":
        return t("retry.reason.transportError");
      case "cycle_timeout":
        return t("retry.reason.cycleTimeout");
      case "empty_final_answer":
        return t("retry.reason.emptyFinalAnswer");
      default:
        return t("retry.reason.transportError");
    }
  })();

  const content = (
    <Text as="p" text03 className="text-sm">
      {t("retry.notice", {
        reason: reasonLabel,
        attempt: notice?.obj.attempt ?? 1,
      })}
    </Text>
  );

  return children([
    {
      icon: SvgRefreshCw,
      status: t("retry.status"),
      content,
      supportsCollapsible: false,
      timelineLayout: "timeline",
      noPaddingRight: true,
    },
  ]);
};
