"""Bucket URLs by service provider (eTLD+1-ish).

Used wherever same-provider bookkeeping is needed without pacing the
requests — currently the known-challenger cooldowns in the crawler and the
browser fetch layer.
"""

from urllib.parse import urlparse

# CDN hosts bucket together with their origin site even though the registrable
# domains differ. Keyed by registrable domain, value is the shared bucket.
_PROVIDER_ALIASES: dict[str, str] = {
    "redd.it": "reddit.com",
    "redditmedia.com": "reddit.com",
    "redditstatic.com": "reddit.com",
    "twitter.com": "x.com",
}

# Two-level public suffixes that would otherwise collapse unrelated sites
# (everything under *.co.uk would share one bucket without this).
_TWO_LEVEL_SUFFIXES: frozenset[str] = frozenset(
    {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "co.jp",
        "or.jp",
        "ne.jp",
        "com.au",
        "net.au",
        "org.au",
        "com.br",
        "com.mx",
        "com.cn",
        "com.tr",
    }
)


def provider_key(url: str) -> str:
    """Bucket a URL by service provider: registrable domain (eTLD+1, with a
    small two-level-suffix list), plus aliases so CDN hosts bucket together
    with the origin (i.redd.it -> reddit.com)."""
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    labels = host.split(".") if host else []
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LEVEL_SUFFIXES:
        registrable = ".".join(labels[-3:])
    elif len(labels) >= 2:
        registrable = ".".join(labels[-2:])
    else:
        registrable = host
    return _PROVIDER_ALIASES.get(registrable, registrable) or "unknown"
