import csv
import io
import zipfile
from io import StringIO

import requests
from temporalio import activity
import pandas as pd


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

SEC_HEADERS = {
    "User-Agent": "unisa-datacollection-2026-07 academic prototype"
}

def get_data_gov_resource_download_url(resource_id: str) -> str:
    api_url = "https://data.gov.au/data/api/3/action/resource_show"

    response = requests.get(
        api_url,
        params={"id": resource_id},
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    return payload["result"]["url"]


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

            primary_organisation = (row.get("Primary Organisation") or "").strip()
            company_name = (row.get("Company Name") or "").strip()

            name = primary_organisation or company_name
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
                    "primary_organisation_name": primary_organisation,
                    "organisation_id": organisation_id,
                    "company_name": company_name,
                    "country": "au",
                    "country_code": "AU",
                    "source_industry_code_type": "ANZSIC",
                    "organisation_name": name,
                    "source_industry_code": None,
                    "source_industry_description": None,
                    "employee_count": None,
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
                if len(data) == 0:
                    print("UK CSV columns:", row.keys())
                if len(data) >= limit:
                    break

                # =============================================
                # Organisation name
                # =============================================

                name = (row.get("CompanyName") or "").strip()

                # =============================================
                # Organisation ID
                # =============================================

                organisation_id = (
                    row.get("CompanyNumber")
                    or row.get(" CompanyNumber")
                    or row.get("Company Number")
                    or row.get("company_number")
                )

                # =============================================
                # Optional filtering
                # =============================================

                if name_keyword and name_keyword.lower() not in name.lower():
                    continue

                # =============================================
                # ETL extraction payload
                # =============================================
                sic_text = row.get("SICCode.SicText_1")
                sic_code = None
                sic_description = None

                if sic_text and " - " in sic_text:
                    sic_code, sic_description = sic_text.split(" - ", 1)
                elif sic_text:
                    sic_code = sic_text
                data.append(
                    {
                        "organisation_name": name,
                        "organisation_id": organisation_id,
                        "country": "uk",
                        "country_code": "UK",
                        "source_industry_code_type": "SIC",
                        "source_industry_code": sic_code,
                        "source_industry_description": sic_description,
                        "employee_count": None,
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
    # =====================================
    # United States
    # US source - SEC EDGAR public company data
    # =====================================================

    elif source == "us_sec":
        tickers_response = requests.get(
            SEC_TICKERS_URL,
            headers=SEC_HEADERS,
            timeout=60,
        )
        tickers_response.raise_for_status()

        tickers_data = tickers_response.json()

        for _, company in tickers_data.items():
            if len(data) >= limit:
                break

            name = (company.get("title") or "").strip()
            cik_raw = company.get("cik_str")
            ticker = company.get("ticker")

            if not name or not cik_raw:
                continue

            if name_keyword and name_keyword.lower() not in name.lower():
                continue

            cik = str(cik_raw).zfill(10)

            sic_code = None
            sic_description = None
            website = None

            try:
                submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)

                submissions_response = requests.get(
                    submissions_url,
                    headers=SEC_HEADERS,
                    timeout=30,
                )

                if submissions_response.status_code == 200:
                    submissions_data = submissions_response.json()

                    sic_code = submissions_data.get("sic")
                    sic_description = submissions_data.get("sicDescription")
                    website = submissions_data.get("website")

            except requests.RequestException:
                pass

            data.append(
                {
                    "organisation_name": name,
                    "organisation_id": cik,
                    "country": "us",
                    "country_code": "US",

                    "source_industry_code_type": "SIC",
                    "source_industry_code": sic_code,
                    "source_industry_description": sic_description,

                    "website": website,
                    "sustainability_url": None,
                    "partner_type": None,
                    "category": None,
                    "organisation_size": None,
                    "employee_count": None,
                    "email": None,
                    "city": None,
                    "state": None,

                    "source": "sec_edgar",
                    "source_url": SEC_SUBMISSIONS_URL.format(cik=cik),
                    "ticker": ticker,
                }
            )
    # =====================================================
    # AU WGEA source - organisation size + ANZSIC data
    # =====================================================

    elif source == "au_wgea":
        resource_id = "4f716314-5de2-425b-aef2-6501c0be076f"

        url = get_data_gov_resource_download_url(resource_id)

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # WGEA resource downloads as a ZIP file containing CSV(s)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        csv_filename = next(
            name for name in zip_file.namelist()
            if name.lower().endswith(".csv")
        )

        with zip_file.open(csv_filename) as csv_file:
            df = pd.read_csv(
                csv_file,
                dtype=str,
                engine="python",
                encoding="latin1",
                on_bad_lines="skip",
            )

        print("WGEA file:", csv_filename, flush=True)
        print("WGEA columns:", list(df.columns), flush=True)
        print("WGEA first row:", df.head(1).to_dict("records"), flush=True)

        seen_abns = set()

        for _, row in df.iterrows():
            if len(data) >= limit:
                break

            organisation_id = row.get("primary_abn")
            name = row.get("primary_employer_name")

            if not name:
                continue

            name = str(name).strip()

            if not organisation_id:
                continue

            organisation_id = str(organisation_id).strip()

            # Avoid repeated questionnaire rows for the same employer
            if organisation_id in seen_abns:
                continue

            seen_abns.add(organisation_id)

            if name_keyword and name_keyword.lower() not in name.lower():
                continue

            anzsic_code = row.get("primary_anzsic")

            anzsic_description = (
                row.get("primary_class_name")
                or row.get("primary_group_name")
                or row.get("primary_subdivision_name")
                or row.get("primary_division_name")
            )

            organisation_size = row.get("submission_group_size")

            data.append(
                {
                    "organisation_name": name,
                    "primary_organisation_name": name,
                    "company name": None,
                    "organisation_id": organisation_id,
                    "country": "au",
                    "country_code": "AU",
                    "organisation_name": name,
                    "source_industry_code_type": "ANZSIC",
                    "source_industry_code": anzsic_code,
                    "source_industry_description": anzsic_description,

                    "website": None,
                    "sustainability_url": None,
                    "partner_type": None,
                    "category": None,
                    "organisation_size": organisation_size,
                    "employee_count": None,
                    "email": None,
                    "city": None,
                    "state": None,

                    "source": "data.gov.au_wgea",
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
