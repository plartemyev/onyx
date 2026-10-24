"""Response policy for stored file bytes served from the app origin.

Uploads keep the MIME type the client declared, so serving one inline is a
stored-XSS primitive unless the type is inert: a text/html or image/svg+xml body
rendered inline runs script on the app origin with the viewer's session.
"""

INLINE_SAFE_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpg",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)

# Media types the browser plays natively (<audio>/<video>). The media stack
# never executes script content, and nosniff keeps mislabeled HTML inert, so
# serving these inline is safe. Enumerated explicitly rather than prefix-matched
# to keep the inline surface reviewed.
INLINE_SAFE_AUDIO_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/mpeg",
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/vnd.wave",
        "audio/ogg",
        "audio/webm",
        "audio/aac",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/flac",
        "audio/x-flac",
        "audio/opus",
    }
)

INLINE_SAFE_VIDEO_MIME_TYPES: frozenset[str] = frozenset(
    {
        "video/mp4",
        "video/webm",
        "video/ogg",
        "video/quicktime",
        "video/x-matroska",
        "video/x-msvideo",
        "video/mpeg",
    }
)

INLINE_SAFE_MIME_TYPES: frozenset[str] = (
    INLINE_SAFE_IMAGE_MIME_TYPES
    | INLINE_SAFE_AUDIO_MIME_TYPES
    | INLINE_SAFE_VIDEO_MIME_TYPES
    | frozenset(
        {
            "application/pdf",
            "text/plain",
        }
    )
)

ATTACHMENT_MEDIA_TYPE: str = "application/octet-stream"

# Version of the policy `resolve_inline_disposition` applies. Endpoints that
# cache their responses mix this into the ETag, so bump it whenever the headers
# or the allowlist change: that is the only way to make clients drop entries
# cached under the old policy.
RESPONSE_POLICY_VERSION: str = "v3"


def resolve_inline_disposition(
    media_type: str,
    *,
    inline_types: frozenset[str] = INLINE_SAFE_MIME_TYPES,
    fallback_media_type: str = ATTACHMENT_MEDIA_TYPE,
    fallback_disposition: str | None = "attachment",
    sandbox: bool = True,
) -> tuple[str, dict[str, str]]:
    """Decide how a stored media type may be served.

    Returns the media type to put on the wire and the security headers to merge
    into the response. A type inside `inline_types` is served as stored; anything
    else is downgraded to `fallback_media_type` and carries
    `fallback_disposition` (pass `None` to omit the header).
    """
    headers: dict[str, str] = {"X-Content-Type-Options": "nosniff"}
    if sandbox:
        headers["Content-Security-Policy"] = "sandbox"

    # Match on the bare type: stored types carry parameters ("image/png;base64").
    if media_type.split(";")[0].strip().lower() in inline_types:
        return media_type, headers

    if fallback_disposition is not None:
        headers["Content-Disposition"] = fallback_disposition
    return fallback_media_type, headers
