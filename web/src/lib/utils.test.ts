import {
  ensureHrefProtocol,
  isFileLinkPlaceholder,
  transformLinkUri,
} from "./utils";

describe("isFileLinkPlaceholder", () => {
  it("matches the bare placeholder and its URL-dressed variants", () => {
    expect(isFileLinkPlaceholder("file_link")).toBe(true);
    expect(isFileLinkPlaceholder("https://file_link")).toBe(true);
    expect(isFileLinkPlaceholder("https://file_link/")).toBe(true);
    expect(isFileLinkPlaceholder("http://file_link/")).toBe(true);
    expect(isFileLinkPlaceholder("  file_link  ")).toBe(true);
  });

  it("does not match real links or prose", () => {
    expect(isFileLinkPlaceholder("/api/chat/file/abc-123")).toBe(false);
    expect(isFileLinkPlaceholder("https://example.com/file_link")).toBe(false);
    expect(isFileLinkPlaceholder("file_linkage")).toBe(false);
    expect(isFileLinkPlaceholder(undefined)).toBe(false);
  });
});

describe("ensureHrefProtocol", () => {
  it("adds https protocol to bare domains", () => {
    expect(ensureHrefProtocol("anthropic.com")).toBe("https://anthropic.com");
  });

  it("preserves links that already include a protocol", () => {
    expect(ensureHrefProtocol("https://anthropic.com")).toBe(
      "https://anthropic.com"
    );
    expect(ensureHrefProtocol("mailto:support@anthropic.com")).toBe(
      "mailto:support@anthropic.com"
    );
  });

  it("converts bare email addresses to mailto links", () => {
    expect(ensureHrefProtocol("support@anthropic.com")).toBe(
      "mailto:support@anthropic.com"
    );
  });
});

describe("transformLinkUri", () => {
  it("allows safe protocols", () => {
    expect(transformLinkUri("https://anthropic.com")).toBe(
      "https://anthropic.com"
    );
    expect(transformLinkUri("mailto:support@anthropic.com")).toBe(
      "mailto:support@anthropic.com"
    );
  });

  it("converts bare email addresses to mailto links", () => {
    expect(transformLinkUri("support@anthropic.com")).toBe(
      "mailto:support@anthropic.com"
    );
  });

  it("blocks unsafe protocols", () => {
    expect(transformLinkUri("javascript:alert(1)")).toBeNull();
  });
});
