# src/temporal/activities/discover_websites.py

import re

import requests
from bs4 import BeautifulSoup
from temporalio import activity

# =========================================================
# Country-specific domain suffixes
# =========================================================

AU_TLDS = [
    ".com.au",
    ".org.au",
    ".com",
    ".org",
]

UK_TLDS = [
    ".co.uk",
    ".org.uk",
    ".uk",
    ".com",
    ".org",
]

DEFAULT_TLDS = [
    ".com",
    ".org",
]


@activity.defn
async def discover_websites(
    extracted_result: dict,
):

    data = extracted_result.get(
        "data",
        [],
    )

    for org in data:
        organisation_name = org.get("organisation_name")

        country_code = org.get("country_code")

        if not organisation_name:
            continue

        website = None

        # =====================================================
        # Clean organisation name
        # =====================================================

        cleaned_name = organisation_name.lower()

        # 🔹 remove company suffixes
        cleaned_name = re.sub(
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
            "",
            cleaned_name,
        )

        # 🔹 remove symbols
        cleaned_name = re.sub(
            r"[^a-z0-9\s]",
            "",
            cleaned_name,
        )

        # 🔹 remove spaces
        domain_base = cleaned_name.strip().replace(" ", "")

        if not domain_base:
            continue

        # =====================================================
        # Select country-specific TLDs
        # =====================================================

        if country_code == "AU":
            tlds = AU_TLDS

        elif country_code == "UK":
            tlds = UK_TLDS

        else:
            tlds = DEFAULT_TLDS

        # =====================================================
        # Generate possible domains
        # =====================================================

        possible_domains = []

        for tld in tlds:
            possible_domains.append(f"https://www.{domain_base}{tld}")

        # =====================================================
        # Validate domains
        # =====================================================

        for candidate_url in possible_domains:
            try:
                response = requests.get(
                    candidate_url,
                    timeout=5,
                    headers={"User-Agent": ("Mozilla/5.0")},
                    allow_redirects=True,
                )

                # =============================================
                # Must return success
                # =============================================

                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(
                    response.text,
                    "html.parser",
                )

                html_text = soup.get_text(
                    separator=" ",
                    strip=True,
                ).lower()

                title = soup.title.string.lower() if soup.title and soup.title.string else ""

                # =============================================
                # Validation logic
                # =============================================

                validation_terms = [
                    organisation_name.lower(),
                    domain_base,
                ]

                matched = any(term in html_text or term in title for term in validation_terms)

                if matched:
                    website = candidate_url

                    break

            except Exception:
                continue

        # =====================================================
        # Save website result
        # =====================================================

        org["website"] = website

    # =========================================================
    # Return ETL payload
    # =========================================================

    return {
        "source": extracted_result.get("source"),
        "requested_limit": extracted_result.get("requested_limit"),
        "returned_count": len(data),
        "sample": data[:5],
        "data": data,
        "note": ("Website discovery with country-specific domain validation completed"),
    }
