const CHAT_FILE_URL_REGEX = /\/api\/chat\/file\/([^/?#]+)/;
const IMAGE_EXTENSIONS = /\.(png|jpe?g|gif|webp|svg|bmp|ico|tiff?)$/i;

export function buildImgUrl(fileId: string) {
  return `/api/chat/file/${fileId}`;
}

/**
 * True when a filename ends with a known image extension.
 */
export function isImageFileName(filename: string): boolean {
  return IMAGE_EXTENSIONS.test(filename);
}

/**
 * If `href` points to a chat file, returns the file ID regardless of the
 * link text. Otherwise returns null.
 */
export function extractChatFileId(href: string | undefined): string | null {
  if (!href) return null;
  const match = CHAT_FILE_URL_REGEX.exec(href);
  return match?.[1] ?? null;
}

/**
 * If `href` points to a chat file and `linkText` ends with an image extension,
 * returns the file ID. Otherwise returns null.
 */
export function extractChatImageFileId(
  href: string | undefined,
  linkText: string
): string | null {
  const fileId = extractChatFileId(href);
  if (!fileId) return null;
  if (!IMAGE_EXTENSIONS.test(linkText)) return null;
  return fileId;
}
