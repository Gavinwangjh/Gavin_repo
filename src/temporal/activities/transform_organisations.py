import re

from temporalio import activity


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


@activity.defn
async def transform_organisations(extracted_data: dict) -> dict:
    records = extracted_data.get("sample", [])

    transformed_records = []

    for record in records:
        transformed_records.append(
            {
                # Categorisation by code
                "country_code": clean_text(record.get("country_code")) or "61",
                "partner_type_code": clean_text(record.get("partner_type_code")) or "2",
                "category_code": clean_text(record.get("category_code")),
                "organisation_size_code": clean_text(
                    record.get("organisation_size_code")
                ),

                # Data ingestion fields
                "partner_type": clean_text(record.get("partner_type")) or "Brand",
                "category_description": clean_text(record.get("category")),
                "organisation_name": clean_text(record.get("organisation_name")),
                "organisation_registration_number": clean_text(
                    record.get("organisation_id")
                ),
                "website_url": clean_url(record.get("website")),
                "sustainability_url": clean_url(record.get("sustainability_url")),
                "city_name": clean_text(record.get("city")),
                "state_name": clean_text(record.get("state")),
                "country_name": "Australia" if clean_text(record.get("country_code")) == "AU" else clean_text(record.get("country")),
                "primary_email_address": clean_email(record.get("email")),
                "organisation_size": clean_text(record.get("organisation_size")),

                # Traceability fields
                "source": clean_text(record.get("source")),
                "source_url": clean_url(record.get("source_url")),
            }
        )

    return {
        "source": extracted_data.get("source"),
        "requested_limit": extracted_data.get("requested_limit"),
        "extracted_count": extracted_data.get("returned_count"),
        "transformed_count": len(transformed_records),
        "transformed_sample": transformed_records,
        "note": "Extracted sample records transformed into required ingestion format.",
    }