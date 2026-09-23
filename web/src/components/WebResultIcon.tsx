"use client";

import { ValidSources } from "@/lib/types";
import { SourceIcon } from "./SourceIcon";
import { SvgOnyxLogo, SvgGithub } from "@opal/logos";

/**
 * Icon for a web result. Everything renders locally: the browser never
 * fetches favicons from the cited host (or a favicon service such as
 * t3.gstatic.com), so reading an agent report cannot leak the user's
 * presence to third parties.
 */
export function WebResultIcon({
  url,
  size = 18,
}: {
  url: string;
  size?: number;
}) {
  let hostname;
  try {
    hostname = new URL(url).hostname;
  } catch (e) {
    hostname = "onyx.app";
  }
  if (hostname.includes("onyx.app")) {
    return <SvgOnyxLogo size={size} className="dark:text-white text-black" />;
  }
  if (hostname === "github.com" || hostname.endsWith(".github.com")) {
    return <SvgGithub size={size} />;
  }
  return <SourceIcon sourceType={ValidSources.Web} iconSize={size} />;
}
