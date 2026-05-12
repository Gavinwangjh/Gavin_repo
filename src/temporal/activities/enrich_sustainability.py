# src/temporal/activities/enrich_sustainability.py


import requests
from bs4 import BeautifulSoup
from temporalio import activity

# =========================================================
# Common sustainability / ESG paths
# =========================================================

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


# =========================================================
# Sustainability keywords
# =========================================================

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


# =========================================================
# PDF report patterns
# =========================================================

PDF_PATTERNS = [
    "sustainability",
    "esg",
    "annual-report",
    "impact-report",
    "climate",
]


@activity.defn
async def enrich_sustainability(enriched_result: dict):

    data = enriched_result.get("data", [])

    for org in data:
        website = org.get("website")

        # =====================================================
        # Skip if no website
        # =====================================================

        if not website:
            continue

        sustainability_found = False

        # =====================================================
        # Try common ESG / sustainability paths
        # =====================================================

        for path in COMMON_PATHS:
            try:
                sustainability_url = website.rstrip("/") + path

                response = requests.get(
                    sustainability_url,
                    timeout=10,
                    allow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0"},
                )

                if response.status_code != 200:
                    continue

                html = response.text.lower()

                # =================================================
                # Keyword scoring
                # =================================================

                keyword_matches = sum(keyword in html for keyword in KEYWORDS)

                # =================================================
                # Parse HTML
                # =================================================

                soup = BeautifulSoup(
                    response.text,
                    "html.parser",
                )

                # =================================================
                # Extract visible text
                # =================================================

                visible_text = soup.get_text(
                    separator=" ",
                    strip=True,
                ).lower()

                text_matches = sum(keyword in visible_text for keyword in KEYWORDS)

                # =================================================
                # Search for ESG / report PDFs
                # =================================================

                pdf_found = False

                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"].lower()

                    if ".pdf" not in href:
                        continue

                    if any(pattern in href for pattern in PDF_PATTERNS):
                        pdf_found = True
                        break

                # =================================================
                # Confidence scoring
                # =================================================

                confidence_score = keyword_matches + text_matches + (5 if pdf_found else 0)

                # =================================================
                # Sustainability page detected
                # =================================================

                if confidence_score >= 5:
                    org["sustainability_url"] = sustainability_url

                    org["sustainability_confidence"] = confidence_score

                    org["sustainability_pdf_detected"] = pdf_found

                    sustainability_found = True

                    break

            except Exception:
                continue

        # =====================================================
        # No sustainability page found
        # =====================================================

        if not sustainability_found:
            org["sustainability_url"] = None

            org["sustainability_confidence"] = 0

            org["sustainability_pdf_detected"] = False

    # =========================================================
    # Return ETL payload
    # =========================================================

    return {
        "source": enriched_result.get("source"),
        "requested_limit": enriched_result.get("requested_limit"),
        "returned_count": len(data),
        # 👇 UI preview
        "sample": data[:5],
        # 👇 actual ETL payload
        "data": data,
        "note": ("Professional sustainability discovery enrichment completed"),
    }
