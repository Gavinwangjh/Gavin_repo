import re

from temporalio import activity


# =========================================================
# Cleaning helpers
# =========================================================

def clean_text(value):
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def clean_email(value):
    value = clean_text(value)

    if value is None:
        return None

    value = value.lower()

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
        return None

    return value


def clean_url(value):
    value = clean_text(value)

    if value is None:
        return None

    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    return value


def first_available(record: dict, keys: list[str]):
    for key in keys:
        value = clean_text(record.get(key))
        if value is not None:
            return value
    return None


def first_available_url(record: dict, keys: list[str]):
    for key in keys:
        value = clean_url(record.get(key))
        if value is not None:
            return value
    return None


def first_available_email(record: dict, keys: list[str]):
    for key in keys:
        value = clean_email(record.get(key))
        if value is not None:
            return value
    return None


# =========================================================
# helpers
# =========================================================

def map_country_code(source: str | None, raw_country_code: str | None):
    source = clean_text(source)
    raw_country_code = clean_text(raw_country_code)

    if source == "au" or raw_country_code == "AU":
        return "61"

    if source == "uk" or raw_country_code == "UK":
        return "44"
    if source == "us_sec" or raw_country_code == "US":
        return "1"

    return raw_country_code


def map_country_name(source: str | None, raw_country: str | None, raw_country_code: str | None):
    source = clean_text(source)
    raw_country = clean_text(raw_country)
    raw_country_code = clean_text(raw_country_code)

    if source == "au" or raw_country_code == "AU":
        return "Australia"

    if source == "uk" or raw_country_code == "UK":
        return "United Kingdom"
    if source == "us_sec" or raw_country_code == "US":
        return "United States"

    return raw_country


def get_source_industry_code_type(source: str | None, record: dict):
    existing_code_type = first_available(
        record,
        [
            "source_industry_code_type",
            "industry_code_type",
            "classification_code_type",
        ],
    )

    if existing_code_type:
        return existing_code_type

    if source == "uk":
        return "SIC"

    if source == "au":
        return "ANZSIC"
    if source == "us_sec":
        return "SIC"

    return None

def map_organisation_size(employee_count):
    if employee_count is None:
        return None, None

    try:
        employee_count = int(employee_count)
    except (TypeError, ValueError):
        return None, None

    if employee_count < 20:
        return "1", "Small"

    if employee_count <= 200:
        return "2", "Medium"

    return "3", "Large"


# =========================================================
# Transform activity
# =========================================================

@activity.defn
async def transform_organisations(extracted_data: dict) -> dict:
    records = extracted_data.get("data") or extracted_data.get("sample", [])

    source = clean_text(extracted_data.get("source"))

    transformed_records = []

    for record in records:
        raw_country_code = first_available(
            record,
            [
                "country_code",
                "countryCode",
            ],
        )

        country_code = map_country_code(source, raw_country_code)

        country_name = map_country_name(
            source,
            first_available(record, ["country", "country_name"]),
            raw_country_code,
        )

        organisation_name = first_available(
            record,
            [
                "organisation_name",
                "organization_name",
                "company_name",
                "Company Name",
                "Primary Organisation",
            ],
        )

        organisation_registration_number = first_available(
            record,
            [
                "organisation_registration_number",
                "organisation_id",
                "registration_number",
                "company_number",
                "abn",
                "ABN",
                "Primary ABN",
                "ï»¿Primary ABN",
            ],
        )

        website_url = first_available_url(
            record,
            [
                "website_url",
                "website",
                "official_website",
                "discovered_website",
            ],
        )

        sustainability_url = first_available_url(
            record,
            [
                "sustainability_url",
                "sustainability_report_url",
                "annual_report_url",
                "sustainability_page",
            ],
        )

        primary_email_address = first_available_email(
            record,
            [
                "primary_email_address",
                "email",
                "contact_email",
            ],
        )

        city_name = first_available(
            record,
            [
                "city_name",
                "city",
                "suburb",
            ],
        )

        state_name = first_available(
            record,
            [
                "state_name",
                "state",
                "region",
            ],
        )

        source_industry_code = first_available(
            record,
            [
                "source_industry_code",
                "industry_code",
                "sic_code",
                "sic_codes",
                "anzsic_code",
                "classification_code",
            ],
        )

        source_industry_description = first_available(
            record,
            [
                "source_industry_description",
                "industry_description",
                "sic_description",
                "anzsic_description",
                "classification_description",
            ],
        )

        source_industry_code_type = get_source_industry_code_type(source, record)

        category_code = clean_text(record.get("category_code"))
        category_description = first_available(
            record,
            [
                "category_description",
                "category",
            ],
        )

        category_mapping_status = (
            "mapped"
            if category_code is not None
            else "pending_mapping"
            if source_industry_code is not None
            else "missing_source_industry_code"
        )

        employee_count = first_available(
            record,
            [
                "employee_count",
                "number_of_employees",
                "staff_count",
                "employees",
            ],
        )

        organisation_size_code, organisation_size = map_organisation_size(employee_count)

        transformed_records.append(
            {
                # Categorisation by code
                "country_code": country_code,
                "partner_type_code": clean_text(record.get("partner_type_code")) or "2",
                "category_code": category_code,
                "organisation_size_code": clean_text(record.get("organisation_size_code")) or organisation_size_code,
                "organisation_size": clean_text(record.get("organisation_size")) or organisation_size,
                "employee_count": employee_count,
                

                # Industry classification traceability
                "source_industry_code_type": source_industry_code_type,
                "source_industry_code": source_industry_code,
                "source_industry_description": source_industry_description,
                "category_mapping_status": category_mapping_status,

                # Data ingestion fields
                "partner_type": clean_text(record.get("partner_type")) or "Brand",
                "category_description": category_description,
                "organisation_name": organisation_name,
                "organisation_registration_number": organisation_registration_number,
                "website_url": website_url,
                "sustainability_url": sustainability_url,
                "sustainability_report_url": clean_url(record.get("sustainability_report_url")),
                "sustainability_confidence": record.get("sustainability_confidence"),
                "sustainability_pdf_detected": record.get("sustainability_pdf_detected"),
                "city_name": city_name,
                "state_name": state_name,
                "country_name": country_name,
                "primary_email_address": primary_email_address,

                # Enrichment status
                "website_lookup_status": clean_text(record.get("website_lookup_status")),
                "sustainability_lookup_status": clean_text(
                    record.get("sustainability_lookup_status")
                ),

                # Traceability fields
                "source": clean_text(record.get("source")),
                "source_url": clean_url(record.get("source_url")),
            }
        )

    return {
        "source": source,
        "requested_limit": extracted_data.get("requested_limit"),
        "extracted_count": extracted_data.get("returned_count"),
        "transformed_count": len(transformed_records),
        "transformed_sample": transformed_records,
        "note": (
            "Extracted and enriched organisation records transformed into "
            "required ingestion format."
        ),
    }