import React from "react";
import { cn } from "@opal/utils";

/** Formats a byte count as KiB, rounded down to 2 decimals ("45.23"). */
export const formatKib = (bytes: number): string =>
  String(Math.floor((bytes * 100) / 1024) / 100);

/** Formats a token count: below 1000 as a plain number, otherwise thousands
 * with a k suffix, rounded down to 2 decimals ("11.37k"). */
export const formatTokenCount = (tokens: number): string => {
  if (tokens < 1000) return String(Math.floor(tokens));
  return `${Math.floor(tokens / 10) / 100}k`;
};

/** Formats a byte count as KiB/MiB/GiB (binary units), rounded down to
 * 2 decimals ("60.01 KiB", "1.2 MiB"). */
export const formatDataSize = (bytes: number): string => {
  if (bytes < 1024 * 1024) {
    return `${Math.floor((bytes * 100) / 1024) / 100} KiB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${Math.floor((bytes * 100) / (1024 * 1024)) / 100} MiB`;
  }
  return `${Math.floor((bytes * 100) / (1024 * 1024 * 1024)) / 100} GiB`;
};

export interface StepHeaderSummaryProps {
  /** Main step label, e.g. "Searching the web" */
  label: string;
  /** Small stat values appended after the label, e.g. ["3 queries", "15 results"] */
  stats: string[];
  className?: string;
}

/** Stats row without the label; dot separators start from the second stat. */
export function StatsRow({ stats }: { stats: string[] }) {
  return (
    <>
      {stats.map((stat, index) => (
        <span
          key={index}
          className={cn(
            "text-xs text-text-03",
            index > 0 && "before:me-1.5 before:content-['·']"
          )}
        >
          {stat}
        </span>
      ))}
    </>
  );
}

/**
 * Step header line: the label followed by small muted stats, each prefixed
 * with a dot separator drawn via CSS (keeps raw text nodes out of JSX).
 * Renders as a span so it can sit inside a Text header or stand alone.
 */
export function StepHeaderSummary({
  label,
  stats,
  className,
}: StepHeaderSummaryProps) {
  if (stats.length === 0) {
    return <span className={className}>{label}</span>;
  }
  return (
    <span className={cn("flex min-w-0 items-center", className)}>
      <span className="truncate">{label}</span>
      {stats.map((stat, index) => (
        <span
          key={index}
          className="shrink-0 before:me-1.5 before:content-['·']"
        >
          <StatsRow stats={[stat]} />
        </span>
      ))}
    </span>
  );
}

export default StepHeaderSummary;
