import {
  buildImgUrl,
  extractChatFileId,
  extractChatImageFileId,
  isImageFileName,
} from "../utils";

describe("chat file url helpers", () => {
  describe("extractChatFileId", () => {
    it("extracts the file id from a relative chat file url", () => {
      expect(extractChatFileId("/api/chat/file/abc-123")).toBe("abc-123");
    });

    it("extracts the file id from an absolute chat file url", () => {
      expect(
        extractChatFileId("https://hal.example.com/api/chat/file/abc-123")
      ).toBe("abc-123");
    });

    it("returns null for non chat file urls and empty input", () => {
      expect(
        extractChatFileId("https://example.com/other/image.png")
      ).toBeNull();
      expect(extractChatFileId("file_link")).toBeNull();
      expect(extractChatFileId(undefined)).toBeNull();
    });
  });

  describe("extractChatImageFileId", () => {
    it("requires an image extension in the link text", () => {
      expect(
        extractChatImageFileId("/api/chat/file/abc-123", "chart.png")
      ).toBe("abc-123");
      expect(
        extractChatImageFileId("/api/chat/file/abc-123", "see the chart")
      ).toBeNull();
    });
  });

  describe("isImageFileName / buildImgUrl", () => {
    it("detects image filenames case-insensitively", () => {
      expect(isImageFileName("photo.JPG")).toBe(true);
      expect(isImageFileName("chart.png")).toBe(true);
      expect(isImageFileName("data.csv")).toBe(false);
    });

    it("builds relative chat file urls", () => {
      expect(buildImgUrl("abc-123")).toBe("/api/chat/file/abc-123");
    });
  });
});
