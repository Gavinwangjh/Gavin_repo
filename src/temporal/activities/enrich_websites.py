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

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}

CONTACT_PATHS = [
    "",
    "/contact",
    "/contact-us",
    "/about",
    "/about-us",
    "/locations",
]

AU_STATES = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]


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


def fetch_page(url):
    try:
        response = requests.get(
            url,
            timeout=20,
            allow_redirects=True,
            headers=REQUEST_HEADERS,
        )

        if response.status_code >= 400:
            return None

        return response.text

    except requests.RequestException:
        return None


def extract_email(text):
    if not text:
        return None

    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)

    if not match:
        return None

    return match.group(0).lower()


def extract_state(text):
    if not text:
        return None

    for state in AU_STATES:
        if re.search(rf"\b{state}\b", text):
            return state

    return None


def extract_contact_details(website_url):
    result = {
        "primary_email_address": None,
        "city": None,
        "state": None,
        "contact_lookup_status": "not_started",
    }

    website_url = clean_url(website_url)

    if not website_url:
        result["contact_lookup_status"] = "skipped_no_website"
        return result

    result["contact_lookup_status"] = "not_found"

    for path in CONTACT_PATHS:
        page_url = urljoin(website_url + "/", path.lstrip("/"))
        html = fetch_page(page_url)

        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        email = extract_email(text)
        state = extract_state(text)

        if email and not result["primary_email_address"]:
            result["primary_email_address"] = email

        if state and not result["state"]:
            result["state"] = state

        if result["primary_email_address"] or result["state"]:
            result["contact_lookup_status"] = "found_contact_details"

        if result["primary_email_address"] and result["state"]:
            break

    return result


def build_dfat_ngo_map():
    html = fetch_page(DFAT_URL)

    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    ngo_map = {}

    sections = {
        "Full Accreditation": "Full accreditation",
        "Base Accreditation": "Base accreditation",
    }

    for category, heading_text in sections.items():
        heading = soup.find(
            lambda tag, heading_text=heading_text: tag.name in ["h2", "h3"] and heading_text in tag.get_text()
        )

        if not heading:
            continue

        ngo_list = heading.find_next("ul")

        if not ngo_list:
            continue

        for li in ngo_list.find_all("li"):
            link = li.find("a")

            if not link:
                continue

            name = clean_text(link.get_text())
            website = clean_url(link.get("href"))

            if not name or not website:
                continue

            ngo_map[name.lower()] = {
                "website": website,
                "partner_type": "NGO",
                "category": category,
            }

    return ngo_map


@activity.defn
async def enrich_websites(extracted_result: dict):
    records = extracted_result.get("data", [])

    try:
        ngo_map = build_dfat_ngo_map()
    except Exception:
        ngo_map = {}

    enriched_records = []

    for record in records:
        org_name = clean_text(record.get("organisation_name"))
        org_key = org_name.lower() if org_name else None

        existing_website = clean_url(
            record.get("website")
            or record.get("website_url")
            or record.get("official_website")
            or record.get("discovered_website")
        )

        website_lookup_status = "not_found"

        # 1. Keep existing website if another step already found it
        if existing_website:
            record["website"] = existing_website
            website_lookup_status = "website_already_available"

        # 2. If no website exists, try DFAT NGO lookup
        elif org_key and org_key in ngo_map:
            ngo_data = ngo_map[org_key]

            record["website"] = ngo_data.get("website")
            record["partner_type"] = ngo_data.get("partner_type")
            record["category"] = ngo_data.get("category")

            website_lookup_status = "matched_dfat_ngo_list"

        else:
            record["website"] = None

        record["website_lookup_status"] = website_lookup_status

        # 3. Extract contact details from website
        contact_details = extract_contact_details(record.get("website"))

        record["primary_email_address"] = (
            clean_text(record.get("primary_email_address"))
            or clean_text(record.get("email"))
            or contact_details.get("primary_email_address")
        )

        record["city"] = (
            clean_text(record.get("city"))
            or clean_text(record.get("city_name"))
            or contact_details.get("city")
        )

        record["state"] = (
            clean_text(record.get("state"))
            or clean_text(record.get("state_name"))
            or contact_details.get("state")
        )

        record["contact_lookup_status"] = contact_details.get("contact_lookup_status")

        enriched_records.append(record)

    return {
        "source": extracted_result.get("source"),
        "requested_limit": extracted_result.get("requested_limit"),
        "returned_count": len(enriched_records),
        "sample": enriched_records[:5],
        "data": enriched_records,
        "note": "Website enrichment completed",
    }
