import csv
import io
import zipfile
from io import StringIO

import requests
from temporalio import activity


@activity.defn
async def extract_organisations(params: dict):

    # =====================================================
    # Dynamic workflow parameters
    # =====================================================

    source = params.get("source")

    limit = params.get("limit", 10)

    name_keyword = params.get("name_keyword")

    data = []

    # =====================================================
    # AU source
    # =====================================================

    if source == "au":
        url = (
            "https://data.gov.au/data/dataset/"
            "4d35cd80-2538-4705-82f3-d0d18e823d98/"
            "resource/2b0cb5e3-05d4-4e95-b9e8-7a1493a01a81/"
            "download/2022_included_organisations_per_abn.csv"
        )

        response = requests.get(url, timeout=60)

        response.raise_for_status()

        csv_file = StringIO(response.text)

        reader = csv.DictReader(csv_file)

        for row in reader:
            if len(data) >= limit:
                break

            # =================================================
            # Organisation name mapping
            # =================================================

            name = (row.get("Primary Organisation") or row.get("Company Name") or "").strip()

            # =================================================
            # Organisation ID mapping
            # =================================================

            organisation_id = row.get("ABN") or row.get("Primary ABN") or row.get("ï»¿Primary ABN")

            # =================================================
            # Optional filtering
            # =================================================

            if name_keyword and name_keyword.lower() not in name.lower():
                continue

            # =================================================
            # ETL extraction payload
            # =================================================

            data.append(
                {
                    "organisation_name": name,
                    "organisation_id": organisation_id,
                    "country": "au",
                    "country_code": "AU",
                    # =========================================
                    # enrichment placeholder fields
                    # =========================================
                    "website": None,
                    "sustainability_url": None,
                    "partner_type": None,
                    "category": None,
                    "organisation_size": None,
                    "email": None,
                    "city": None,
                    "state": None,
                    # =========================================
                    # metadata
                    # =========================================
                    "source": "data.gov.au",
                    "source_url": url,
                }
            )

    # =====================================================
    # UK source
    # =====================================================

    elif source == "uk":
        url = "https://download.companieshouse.gov.uk/BasicCompanyData-2026-05-01-part1_7.zip"

        response = requests.get(url, timeout=60)

        response.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(response.content))

        filename = z.namelist()[0]

        with z.open(filename) as f:
            reader = csv.DictReader(
                line.decode(
                    "utf-8",
                    errors="ignore",
                )
                for line in f
            )

            for row in reader:
                if len(data) >= limit:
                    break

                # =============================================
                # Organisation name
                # =============================================

                name = (row.get("CompanyName") or "").strip()

                # =============================================
                # Organisation ID
                # =============================================

                organisation_id = row.get("CompanyNumber")

                # =============================================
                # Optional filtering
                # =============================================

                if name_keyword and name_keyword.lower() not in name.lower():
                    continue

                # =============================================
                # ETL extraction payload
                # =============================================

                data.append(
                    {
                        "organisation_name": name,
                        "organisation_id": organisation_id,
                        "country": "uk",
                        "country_code": "UK",
                        # =====================================
                        # enrichment placeholder fields
                        # =====================================
                        "website": None,
                        "sustainability_url": None,
                        "partner_type": None,
                        "category": None,
                        "organisation_size": None,
                        "email": None,
                        "city": None,
                        "state": None,
                        # =====================================
                        # metadata
                        # =====================================
                        "source": "companies_house",
                        "source_url": url,
                    }
                )

    # =====================================================
    # Invalid source
    # =====================================================

    else:
        raise ValueError(f"Unknown source: {source}")

    # =====================================================
    # ETL output
    # =====================================================

    return {
        "source": source,
        "requested_limit": limit,
        "returned_count": len(data),
        # 👇 UI preview
        "sample": data[:3],
        # 👇 actual ETL payload
        "data": data,
        "note": ("Data is sampled for performance (POC mode)"),
    }
