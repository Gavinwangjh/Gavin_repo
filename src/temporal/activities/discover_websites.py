import asyncio
import re
from collections.abc import Iterable

import requests
from bs4 import BeautifulSoup
from temporalio import activity

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_TIMEOUT = 5

COMPANY_SUFFIX_RE = re.compile(
    r"\b("
    r"pty ltd|"
    r"pty|"
    r"limited|"
    r"ltd|"
    r"inc|"
    r"llc|"
    r"group|"
    r"holdings|"
    r"corporation|"
    r"corp"
    r")\b",
    flags=re.IGNORECASE,
)

NON_DOMAIN_CHARS_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")

AU_TLDS = (
    ".com.au",
    ".org.au",
    ".com",
    ".org",
)

UK_TLDS = (
    ".co.uk",
    ".org.uk",
    ".uk",
    ".com",
    ".org",
)

DEFAULT_TLDS = (
    ".com",
    ".org",
)


def clean_text(value):
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def clean_url(value):
    value = clean_text(value)

    if value is None:
        return None

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    return value.rstrip("/")


def clean_organisation_name(organisation_name: str) -> str:
    cleaned_name = organisation_name.lower()
    cleaned_name = COMPANY_SUFFIX_RE.sub("", cleaned_name)
    cleaned_name = NON_DOMAIN_CHARS_RE.sub("", cleaned_name)
    cleaned_name = WHITESPACE_RE.sub(" ", cleaned_name)
    return cleaned_name.strip()


def build_domain_base(organisation_name: str) -> str | None:
    cleaned_name = clean_organisation_name(organisation_name)
    domain_base = cleaned_name.replace(" ", "")
    return domain_base or None


def select_tlds(country_code: str | None) -> tuple[str, ...]:
    country_code = (country_code or "").upper()

    if country_code == "AU":
        return AU_TLDS

    if country_code in {"UK", "GB"}:
        return UK_TLDS

    return DEFAULT_TLDS


def dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    unique_values = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        unique_values.append(value)

    return unique_values


def build_candidate_urls(domain_base: str, country_code: str | None) -> list[str]:
    urls = []

    for tld in select_tlds(country_code):
        domain = f"{domain_base}{tld}"
        urls.append(f"https://www.{domain}")
        urls.append(f"https://{domain}")

    return dedupe(urls)


def page_matches_organisation(html: str, title: str, organisation_name: str, domain_base: str) -> bool:
    page_text = f"{title} {html}".lower()
    searchable_text = NON_DOMAIN_CHARS_RE.sub(" ", page_text)
    searchable_text = WHITESPACE_RE.sub(" ", searchable_text)

    cleaned_name = clean_organisation_name(organisation_name)
    validation_terms = dedupe(
        term
        for term in (
            organisation_name.lower(),
            cleaned_name,
            domain_base,
        )
        if term
    )

    return any(term in searchable_text or term in page_text for term in validation_terms)


def fetch_candidate(session: requests.Session, candidate_url: str) -> tuple[str, str] | None:
    try:
        response = session.get(
            candidate_url,
            timeout=REQUEST_TIMEOUT,
            headers=REQUEST_HEADERS,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    html_text = soup.get_text(separator=" ", strip=True)
    title = soup.title.string if soup.title and soup.title.string else ""

    return html_text, title


def discover_website_for_org(org: dict, session: requests.Session) -> str | None:
    existing_website = clean_url(
        org.get("website")
        or org.get("website_url")
        or org.get("official_website")
        or org.get("discovered_website")
    )

    if existing_website:
        org["website_lookup_status"] = "website_already_available"
        return existing_website

    organisation_name = clean_text(org.get("organisation_name"))

    if not organisation_name:
        org["website_lookup_status"] = "skipped_missing_organisation_name"
        return None

    domain_base = build_domain_base(organisation_name)

    if not domain_base:
        org["website_lookup_status"] = "skipped_empty_domain_base"
        return None

    for candidate_url in build_candidate_urls(domain_base, org.get("country_code")):
        page = fetch_candidate(session, candidate_url)

        if not page:
            continue

        html_text, title = page

        if page_matches_organisation(html_text, title, organisation_name, domain_base):
            org["website_lookup_status"] = "discovered_by_domain_guess"
            org["discovered_website"] = candidate_url
            return candidate_url

    org["website_lookup_status"] = "not_found"
    return None


def discover_websites_sync(extracted_result: dict) -> dict:
    data = extracted_result.get("data", [])

    with requests.Session() as session:
        for org in data:
            org["website"] = discover_website_for_org(org, session)

    return {
        "source": extracted_result.get("source"),
        "requested_limit": extracted_result.get("requested_limit"),
        "returned_count": len(data),
        "sample": data[:5],
        "data": data,
        "note": "Website discovery with country-specific domain validation completed",
    }


@activity.defn
async def discover_websites(extracted_result: dict):
    return await asyncio.to_thread(discover_websites_sync, extracted_result)
