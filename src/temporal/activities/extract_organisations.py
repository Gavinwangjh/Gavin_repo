import csv
import io
import zipfile
from io import StringIO

import requests
from temporalio import activity


@activity.defn
async def extract_organisations(source: str, limit: int = 500, name_keyword: str | None = None):
    data = []

    # 🟩 AU 数据源
    if source == "au":
        url = "https://data.gov.au/data/dataset/4d35cd80-2538-4705-82f3-d0d18e823d98/resource/2b0cb5e3-05d4-4e95-b9e8-7a1493a01a81/download/2022_included_organisations_per_abn.csv"

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        csv_file = StringIO(response.text)
        reader = csv.DictReader(csv_file)

        for row in reader:
            if len(data) >= limit:
                break

            name = (row.get("Primary Organisation") or row.get("Company Name") or "").strip()

            # 🔹 filtering（防止 name 是 None）
            if name_keyword and name_keyword.lower() not in name.lower():
                continue

            data.append(
                {
                    "organisation_name": name,
                    "organisation_id": row.get("ABN") or row.get("Primary ABN") or row.get("ï»¿Primary ABN"),
                    "country": "au",
                    "country_code": "AU",
                    # 👇 占位字段
                    "website": None,
                    "sustainability_url": None,
                    "partner_type": None,
                    "category": None,
                    "organisation_size": None,
                    "email": None,
                    "city": None,
                    "state": None,
                    "source": "data.gov.au",
                    "source_url": url,
                }
            )

    # 🟦 UK 数据源
    elif source == "uk":
        url = "https://download.companieshouse.gov.uk/BasicCompanyData-2026-05-01-part1_7.zip"

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(response.content))
        filename = z.namelist()[0]

        with z.open(filename) as f:
            reader = csv.DictReader(line.decode("utf-8", errors="ignore") for line in f)

            for row in reader:
                if len(data) >= limit:
                    break

                name = (row.get("CompanyName") or "").strip()

                # 🔹 filtering
                if name_keyword and name_keyword.lower() not in name.lower():
                    continue

                data.append(
                    {
                        "organisation_name": name,
                        "organisation_id": row.get("CompanyNumber"),
                        "country": "uk",
                        "country_code": "UK",
                        # 👇 占位字段
                        "website": None,
                        "sustainability_url": None,
                        "partner_type": None,
                        "category": None,
                        "organisation_size": None,
                        "email": None,
                        "city": None,
                        "state": None,
                        "source": "companies_house",
                        "source_url": url,
                    }
                )

    else:
        raise ValueError("Unknown source")

    # 🔹 返回结构（UI友好）
    return {
        "source": source,
        "requested_limit": limit,
        "returned_count": len(data),
        "sample": data[:3],
        "note": "Data is sampled for performance (POC mode)",
    }
