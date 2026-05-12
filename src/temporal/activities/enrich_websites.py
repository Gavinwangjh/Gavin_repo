# src/temporal/activities/enrich_websites.py

import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from temporalio import activity


DFAT_URL = (
    "https://www.dfat.gov.au/development/"
    "who-we-work-with/ngos/"
    "list-of-australian-accredited-"
    "non-government-organisations"
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

    if value.startswith("/"):
        return value

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    return value


def extract_email_from_text(text: str):
    if not text:
        return None

    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)

    if match:
        return match.group(0).lower()

    return None


def url_exists(url: str):
    try:
        response = requests.get(url, timeout=15, allow_redirects=True)
        return response.status_code < 400
    except requests.RequestException:
        return False


def fetch_page(url: str):
    try:
        response = requests.get(url, timeout=20, allow_redirects=True)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def find_contact_details(website_url: str):
    result = {
        "primary_email_address": None,
        "city": None,
        "state": None,
    }

    if not website_url:
        return result

    candidate_paths = [
        "",
        "/contact",
        "/contact-us",
        "/about",
        "/about-us",
        "/locations",
    ]

    for path in candidate_paths:
        page_url = urljoin(website_url.rstrip("/") + "/", path.lstrip("/"))
        html = fetch_page(page_url)

        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True)

        email = extract_email_from_text(page_text)

        if email and not result["primary_email_address"]:
            result["primary_email_address"] = email

        # simple AU state detection
        for state in ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]:
            if state in page_text and not result["state"]:
                result["state"] = state

        if result["primary_email_address"] and result["state"]:
            break

    return result


def build_dfat_ngo_map():
    response = requests.get(DFAT_URL, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    ngo_map = {}

    accreditation_sections = {
        "Full Accreditation": "Full accreditation",
        "Base Accreditation": "Base accreditation",
    }

    for category, heading_text in accreditation_sections.items():
        heading = soup.find(
            lambda tag: tag.name in ["h2", "h3"]
            and heading_text in tag.get_text()
        )

        if not heading:
            continue

        ngo_list = heading.find_next("ul")

        if not ngo_list:
            continue

        for li in ngo_list.find_all("li"):
            a = li.find("a")

            if not a:
                continue

            ngo_name = a.get_text(strip=True)
            ngo_url = a.get("href")

            ngo_map[ngo_name.lower()] = {
                "website": clean_url(ngo_url),
                "partner_type": "NGO",
                "category": category,
            }

    return ngo_map


@activity.defn
async def enrich_websites(extracted_result: dict):
    extracted_data = extracted_result.get("data", [])

    try:
        ngo_map = build_dfat_ngo_map()
    except Exception:
        ngo_map = {}

    enriched_data = []

    for org in extracted_data:
        org_name = clean_text(org.get("organisation_name"))
        org_name_key = org_name.lower() if org_name else None

        enrichment = ngo_map.get(org_name_key) if org_name_key else None

        if enrichment:
            org["website"] = enrichment.get("website")
            org["partner_type"] = enrichment.get("partner_type")
            org["category"] = enrichment.get("category")
            org["website_lookup_status"] = "matched_dfat_ngo_list"
        elif org.get("website") or org.get("website_url"):
            org["website_lookup_status"] = "website_already_available"
        else:
            org["website_lookup_status"] = "not_found"

        website = clean_url(org.get("website") or org.get("website_url"))

        if website:
            contact_details = find_contact_details(website)

            org["website"] = website
            org["primary_email_address"] = contact_details.get("primary_email_address")
            org["city"] = contact_details.get("city")
            org["state"] = contact_details.get("state")

        enriched_data.append(org)

    return {
        "source": extracted_result.get("source"),
        "requested_limit": extracted_result.get("requested_limit"),
        "returned_count": len(enriched_data),
        "sample": enriched_data[:5],
        "data": enriched_data,
        "note": "Website enrichment completed",
    }