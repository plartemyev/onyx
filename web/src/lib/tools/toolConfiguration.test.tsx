import { renderHook, act } from "@testing-library/react";
import { useToolConfiguration } from "@/lib/tools/hooks";
import {
  selectedSourcesFrom,
  buildFilters,
  toggleSourceSelection,
  normalizeSourceSelection,
} from "@/lib/searchFilters/utils";
import type { SourceMetadata } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";
import type { MinimalAgent } from "@/lib/agents/types";
import type { ToolSnapshot } from "@/lib/tools/types";
import { SEARCH_TOOL_ID, WEB_SEARCH_TOOL_ID } from "@/lib/tools/constants";

const CHAT_KEY = "onyx:tools:chat:abc";

// Mutable so a test can move the composer between chats; the hook re-keys
// from what useAppPosition reports on each render.
let mockChatId = "abc";

// Mutable so a test can attach the agent whose tools the composer offers.
let mockActiveAgent: MinimalAgent | undefined;

jest.mock("@/lib/position/hooks", () => ({
  useAppPosition: () => ({
    chat: () => mockChatId,
    agent: () => null,
    project: () => null,
  }),
}));

jest.mock("@/lib/agents/hooks", () => ({
  useActiveAgent: () => mockActiveAgent,
}));

beforeEach(() => {
  sessionStorage.clear();
  mockChatId = "abc";
  mockActiveAgent = undefined;
});

function source(internalName: ValidSources, uniqueKey: string): SourceMetadata {
  return { internalName, uniqueKey } as SourceMetadata;
}

const NOTION = source(ValidSources.Notion, "notion");
const SLACK = source(ValidSources.Slack, "slack");

function toolSnapshot(id: number, inCodeToolId: string | null): ToolSnapshot {
  return {
    id,
    name: inCodeToolId ?? `tool_${id}`,
    display_name: inCodeToolId ?? `Tool ${id}`,
    description: "",
    definition: null,
    custom_headers: [],
    in_code_tool_id: inCodeToolId,
    passthrough_auth: false,
    enabled: true,
    chat_selectable: true,
    agent_creation_selectable: true,
    default_enabled: false,
  };
}

function agentWithTools(tools: ToolSnapshot[]): MinimalAgent {
  return {
    id: 0,
    name: "Default",
    description: "",
    tools,
    starter_messages: null,
    document_sets: [],
    is_public: false,
    is_listed: false,
    display_priority: null,
    is_featured: false,
    builtin_persona: true,
    owner: null,
    owner_group: null,
    user_permission: null,
  };
}

describe("useToolConfiguration — stored shapes", () => {
  test("the pre-filters flat shape still reads as the tool map", () => {
    sessionStorage.setItem(
      CHAT_KEY,
      JSON.stringify({ 1: "forced", 2: "disabled" })
    );
    const { result } = renderHook(() => useToolConfiguration());

    expect(result.current.forcedToolId).toBe(1);
    expect(result.current.disabledToolIds).toEqual([2]);
    expect(result.current.filters.selectedSources).toBeNull();
    expect(result.current.filters.timeRange).toBeNull();
  });

  test("filters ride the snapshot through storage", () => {
    const { result, unmount } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: ["notion"],
      }));
    });
    unmount();

    const { result: reread } = renderHook(() => useToolConfiguration());
    expect(reread.current.filters.selectedSources).toEqual(["notion"]);
  });
});

describe("useToolConfiguration — chat scoping", () => {
  test("each chat keeps its own configuration", () => {
    const { result, rerender } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: ["notion"],
      }));
    });
    act(() => result.current.toggleToolState(7, "forced"));

    // Another chat reads its own (neutral) configuration...
    mockChatId = "other";
    rerender();
    expect(result.current.filters.selectedSources).toBeNull();
    expect(result.current.forcedToolId).toBeNull();

    // ...and edits it without touching the first chat's.
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: [],
      }));
    });

    mockChatId = "abc";
    rerender();
    expect(result.current.filters.selectedSources).toEqual(["notion"]);
    expect(result.current.forcedToolId).toBe(7);
  });
});

describe("useToolConfiguration — orthogonal axes", () => {
  test("editing sources never touches a forced pin", () => {
    const { result } = renderHook(() => useToolConfiguration());
    act(() => result.current.toggleToolState(7, "forced"));
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: [],
      }));
    });
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: null,
      }));
    });

    expect(result.current.forcedToolId).toBe(7);
  });

  test("tool state changes never touch the source selection", () => {
    const { result } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: ["notion"],
      }));
    });
    act(() => result.current.toggleToolState(7, "forced"));
    act(() => result.current.toggleToolState(7, "forced"));
    act(() => result.current.toggleToolState(8, "disabled"));

    expect(result.current.filters.selectedSources).toEqual(["notion"]);
  });
});

describe("useToolConfiguration — internal search starts off", () => {
  test("a fresh chat reads the search tool as disabled and leaves the rest alone", () => {
    mockActiveAgent = agentWithTools([
      toolSnapshot(3, SEARCH_TOOL_ID),
      toolSnapshot(4, WEB_SEARCH_TOOL_ID),
    ]);
    const { result } = renderHook(() => useToolConfiguration());

    expect(result.current.disabledToolIds).toEqual([3]);
    expect(result.current.forcedToolId).toBeNull();
    expect(result.current.stateOf(4)).toBeNull();
  });

  test("no search tool on the agent means no default", () => {
    mockActiveAgent = agentWithTools([toolSnapshot(4, WEB_SEARCH_TOOL_ID)]);
    const { result } = renderHook(() => useToolConfiguration());

    expect(result.current.disabledToolIds).toEqual([]);
  });

  test("pinning search overrides the default and releases back into it", () => {
    mockActiveAgent = agentWithTools([toolSnapshot(3, SEARCH_TOOL_ID)]);
    const { result } = renderHook(() => useToolConfiguration());
    act(() => result.current.toggleToolState(3, "forced"));
    expect(result.current.forcedToolId).toBe(3);
    expect(result.current.disabledToolIds).toEqual([]);

    act(() => result.current.toggleToolState(3, "forced"));
    expect(result.current.forcedToolId).toBeNull();
    expect(result.current.disabledToolIds).toEqual([3]);
  });

  test("an explicit disabled entry for the search tool is not doubled", () => {
    sessionStorage.setItem(
      CHAT_KEY,
      JSON.stringify({ tools: { 3: "disabled" } })
    );
    mockActiveAgent = agentWithTools([toolSnapshot(3, SEARCH_TOOL_ID)]);
    const { result } = renderHook(() => useToolConfiguration());

    expect(result.current.disabledToolIds).toEqual([3]);
  });

  test("the seed rides along when the composer hands off to a new chat", () => {
    mockActiveAgent = agentWithTools([toolSnapshot(3, SEARCH_TOOL_ID)]);
    const { result } = renderHook(() => useToolConfiguration());
    act(() => result.current.handOffTo("chat-xyz"));

    const stored = JSON.parse(
      sessionStorage.getItem("onyx:tools:chat:chat-xyz") ?? "null"
    );
    expect(stored.tools).toEqual({ 3: "disabled" });
  });
});

describe("selectedSourcesFrom — positive-selection semantics", () => {
  const configured = [NOTION, SLACK];

  test("an untouched selection means no source filter at all", () => {
    const { result } = renderHook(() => useToolConfiguration());
    const selected = selectedSourcesFrom(result.current.filters, configured);

    expect(selected).toBeNull();
    expect(buildFilters(selected, [], null, []).source_type).toBeNull();
  });

  test("an explicit selection narrows to those sources", () => {
    const { result } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: ["notion"],
      }));
    });
    const selected = selectedSourcesFrom(result.current.filters, configured);

    expect(selected).toEqual([NOTION]);
    expect(buildFilters(selected, [], null, []).source_type).toEqual([
      ValidSources.Notion,
    ]);
  });

  test("selecting nothing sends an explicitly empty list", () => {
    const { result } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: [],
      }));
    });
    const selected = selectedSourcesFrom(result.current.filters, configured);

    expect(selected).toEqual([]);
    expect(buildFilters(selected, [], null, []).source_type).toEqual([]);
  });

  test("a selection naming a removed connector drops it", () => {
    const { result } = renderHook(() => useToolConfiguration());
    act(() => {
      result.current.setFilters((current) => ({
        ...current,
        selectedSources: ["notion", "gone"],
      }));
    });
    const selected = selectedSourcesFrom(result.current.filters, configured);

    expect(selected).toEqual([NOTION]);
  });
});

describe("toggleSourceSelection — edit-boundary normalization", () => {
  const keys = ["notion", "slack"];

  test("the first toggle materialises the untouched default", () => {
    expect(toggleSourceSelection(null, "slack", keys)).toEqual(["notion"]);
  });

  test("re-covering every configured source collapses to the sentinel", () => {
    expect(toggleSourceSelection(["notion"], "slack", keys)).toBeNull();
  });

  test("toggling the last source off leaves the explicit empty set", () => {
    expect(toggleSourceSelection(["notion"], "notion", keys)).toEqual([]);
  });

  test("a covering write from any surface collapses to the sentinel", () => {
    expect(normalizeSourceSelection(["slack", "notion"], keys)).toBeNull();
    expect(normalizeSourceSelection(["notion"], keys)).toEqual(["notion"]);
    expect(normalizeSourceSelection([], keys)).toEqual([]);
  });
});
