# src/temporal/activities/enrich_websites.py

import requests
from bs4 import BeautifulSoup
from temporalio import activity

DFAT_URL = (
    "https://www.dfat.gov.au/development/"
    "who-we-work-with/ngos/"
    "list-of-australian-accredited-"
    "non-government-organisations"
)


@activity.defn
async def enrich_websites(extracted_result: dict):

    extracted_data = extracted_result.get("data", [])

    # 🔹 获取 DFAT NGO 页面
    response = requests.get(DFAT_URL, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # 🔹 NGO enrichment map
    ngo_map = {}

    # =========================================================
    # 🟢 Full accreditation NGOs
    # =========================================================

    full_heading = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Full accreditation" in tag.get_text())

    if full_heading:
        full_list = full_heading.find_next("ul")

        if full_list:
            for li in full_list.find_all("li"):
                a = li.find("a")

                if not a:
                    continue

                ngo_name = a.get_text(strip=True)
                ngo_url = a.get("href")

                ngo_map[ngo_name.lower()] = {
                    "website": ngo_url,
                    "partner_type": "NGO",
                    "category": "Full Accreditation",
                }

    # =========================================================
    # 🟡 Base accreditation NGOs
    # =========================================================

    base_heading = soup.find(lambda tag: tag.name in ["h2", "h3"] and "Base accreditation" in tag.get_text())

    if base_heading:
        base_list = base_heading.find_next("ul")

        if base_list:
            for li in base_list.find_all("li"):
                a = li.find("a")

                if not a:
                    continue

                ngo_name = a.get_text(strip=True)
                ngo_url = a.get("href")

                ngo_map[ngo_name.lower()] = {
                    "website": ngo_url,
                    "partner_type": "NGO",
                    "category": "Base Accreditation",
                }

    # =========================================================
    # 🔹 Merge enrichment into extracted data
    # =========================================================

    enriched_data = []

    for org in extracted_data:
        org_name = org.get("organisation_name", "").strip().lower()

        enrichment = ngo_map.get(org_name)

        # 🟢 Match found
        if enrichment:
            org["website"] = enrichment["website"]
            org["partner_type"] = enrichment["partner_type"]
            org["category"] = enrichment["category"]

            # 🔹 placeholder sustainability URL
            if org["website"]:
                org["sustainability_url"] = org["website"].rstrip("/") + "/sustainability"

        enriched_data.append(org)

    # =========================================================
    # 🔹 Return ETL payload
    # =========================================================

    return {
        "source": extracted_result.get("source"),
        "requested_limit": extracted_result.get("requested_limit"),
        "returned_count": len(enriched_data),
        # 👇 UI preview
        "sample": enriched_data[:5],
        # 👇 actual ETL payload
        "data": enriched_data,
        "note": "DFAT NGO enrichment completed",
    }
