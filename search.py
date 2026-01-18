import requests
from urllib.parse import urlparse
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID, BLACKLIST_DOMAINS, WHITELIST_DOMAINS

from urllib.parse import urlparse
from config import BLACKLIST_DOMAINS

def domain_allowed(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    # Only block blacklisted domains
    if any(b in domain for b in BLACKLIST_DOMAINS):
        return False
    return True

# (google_search function remains the same, but now it will actually return URLs)


def google_search(query: str, num_results: int):
    urls = []
    start = 1

    while len(urls) < num_results:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "start": start,
            },
            timeout=10
        )

        data = resp.json()
        items = data.get("items", [])

        for item in items:
            url = item["link"]
            if domain_allowed(url):
                urls.append(url)
                if len(urls) >= num_results:
                    break

        start += 10
        if not items:
            break

    return urls
