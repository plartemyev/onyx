import pytest

from onyx.utils.provider_key import provider_key


@pytest.mark.parametrize(
    "url, expected",
    [
        # Image CDN hosts group with their origin
        ("https://i.imgur.com/AKVZMd3b.jpg", "imgur.com"),
        ("https://imgur.com/trending", "imgur.com"),
        ("https://i.redd.it/abc.jpg", "reddit.com"),
        ("https://preview.redd.it/abc.jpg?auto=webp", "reddit.com"),
        ("https://external-preview.redd.it/abc", "reddit.com"),
        ("https://www.reddit.com/r/pics/", "reddit.com"),
        # x.com and twitter.com are one provider
        ("https://x.com/user/status/1", "x.com"),
        ("https://twitter.com/user/status/1", "x.com"),
        # Plain registrable domains
        ("https://example.com/cat.jpg", "example.com"),
        ("https://api.example.com/v1/file", "example.com"),
        # Two-level public suffixes keep distinct sites distinct
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://www.mit.edu/", "mit.edu"),
        # Scheme-relative / junk still buckets somewhere stable
        ("//cdn.example.org/a.png", "example.org"),
    ],
)
def test_provider_key(url: str, expected: str) -> None:
    assert provider_key(url) == expected
