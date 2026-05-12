# src/temporal/activities/enrich_sustainability.py

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from temporalio import activity


COMMON_PATHS = [
    "/sustainability",
    "/sustainability/",
    "/esg",
    "/esg/",
    "/annual-report",
    "/annual-reports",
    "/annual-report/",
    "/annual-reports/",
    "/impact",
    "/impact/",
    "/our-impact",
    "/our-impact/",
    "/responsibility",
    "/responsibility/",
    "/sustainability-report",
    "/sustainability-report/",
    "/environment",
    "/climate",
    "/climate-action",
]


KEYWORDS = [
    "sustainability",
    "sustainable",
    "esg",
    "environment",
    "climate",
    "carbon",
    "net zero",
    "social impact",
    "governance",
    "annual report",
    "impact report",
    "sustainability report",
    "emissions",
    "renewable",
    "csr",
]


PDF_PATTERNS = [
    "sustainability",
    "esg",
    "annual-report",
    "annual_report",
    "impact-report",
    "impact_report",
    "climate",
]


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

    return value


def fetch_page(url: str):
    try:
        response = requests.get(
            url,
            timeout=15,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )

        if response.status_code >= 400:
            return None, None

        return response.text, response.url

    except requests.RequestException:
        return None, None


def score_page(html: str):
    html_lower = html.lower()

    soup = BeautifulSoup(html, "html.parser")
    visible_text = soup.get_text(separator=" ", strip=True).lower()

    keyword_matches = sum(keyword in html_lower for keyword in KEYWORDS)
    text_matches = sum(keyword in visible_text for keyword in KEYWORDS)

    return keyword_matches + text_matches, soup


def find_report_pdf(soup: BeautifulSoup, page_url: str):
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        href_lower = href.lower()

        if ".pdf" not in href_lower:
            continue

        if any(pattern in href_lower for pattern in PDF_PATTERNS):
            return urljoin(page_url, href)

    return None


@activity.defn
async def enrich_sustainability(enriched_result: dict):
    data = enriched_result.get("data", [])

    for org in data:
        website = clean_url(org.get("website") or org.get("website_url"))

        if not website:
            org["sustainability_url"] = None
            org["sustainability_report_url"] = None
            org["sustainability_confidence"] = 0
            org["sustainability_pdf_detected"] = False
            org["sustainability_lookup_status"] = "skipped_no_website"
            continue

        best_match = {
            "url": None,
            "report_url": None,
            "confidence": 0,
            "pdf_detected": False,
            "status": "not_found",
        }

        for path in COMMON_PATHS:
            candidate_url = urljoin(website.rstrip("/") + "/", path.lstrip("/"))

            html, final_url = fetch_page(candidate_url)

            if not html:
                continue

            confidence_score, soup = score_page(html)
            report_pdf_url = find_report_pdf(soup, final_url)

            if report_pdf_url:
                confidence_score += 5

            if confidence_score > best_match["confidence"]:
                best_match = {
                    "url": final_url,
                    "report_url": report_pdf_url,
                    "confidence": confidence_score,
                    "pdf_detected": report_pdf_url is not None,
                    "status": "found_candidate",
                }

            if confidence_score >= 5:
                break

        org["sustainability_url"] = best_match["url"]
        org["sustainability_report_url"] = best_match["report_url"]
        org["sustainability_confidence"] = best_match["confidence"]
        org["sustainability_pdf_detected"] = best_match["pdf_detected"]
        org["sustainability_lookup_status"] = best_match["status"]

    return {
        "source": enriched_result.get("source"),
        "requested_limit": enriched_result.get("requested_limit"),
        "returned_count": len(data),
        "sample": data[:5],
        "data": data,
        "note": "Sustainability discovery enrichment completed",
    }