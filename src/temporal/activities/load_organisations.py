import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from temporalio import activity


def clean_text(value):
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def safe_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def get_records(transformed_data: dict):
    return (
        transformed_data.get("transformed_records")
        or transformed_data.get("transformed_sample")
        or []
    )


async def lookup_country_id(conn, country_code):
    country_code = safe_int(country_code)

    if country_code is None:
        return None

    result = await conn.execute(
        text(
            """
            SELECT "Id"
            FROM "Countries"
            WHERE "CountryCode" = :country_code
            """
        ),
        {"country_code": country_code},
    )

    row = result.fetchone()
    return row[0] if row else None


async def valid_partner_type(conn, partner_type_code):
    partner_type_code = safe_int(partner_type_code)

    if partner_type_code is None:
        return None

    result = await conn.execute(
        text(
            """
            SELECT 1
            FROM "PartnerType"
            WHERE "PartnerTypeCode" = :partner_type_code
            """
        ),
        {"partner_type_code": partner_type_code},
    )

    return partner_type_code if result.fetchone() else None


async def valid_category(conn, category_code):
    category_code = safe_int(category_code)

    if category_code is None:
        return None

    result = await conn.execute(
        text(
            """
            SELECT 1
            FROM "CategoryTypes"
            WHERE "CategoryCode" = :category_code
            """
        ),
        {"category_code": category_code},
    )

    return category_code if result.fetchone() else None


async def valid_organisation_size(conn, organisation_size_code):
    organisation_size_code = safe_int(organisation_size_code)

    if organisation_size_code is None:
        return None

    result = await conn.execute(
        text(
            """
            SELECT 1
            FROM "OrganisationSize"
            WHERE "OrganisationSizeCode" = :organisation_size_code
            """
        ),
        {"organisation_size_code": organisation_size_code},
    )

    return organisation_size_code if result.fetchone() else None


@activity.defn
async def load_organisations(transformed_data: dict) -> dict:
    records = get_records(transformed_data)

    if not records:
        return {
            "received": 0,
            "inserted": 0,
            "skipped": 0,
            "status": "no_data",
        }

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        return {
            "received": len(records),
            "inserted": 0,
            "skipped": len(records),
            "status": "missing_database_url",
        }

    engine = create_async_engine(database_url)

    inserted = 0
    skipped = 0

    async with engine.begin() as conn:
        for record in records:
            organisation_name = clean_text(record.get("organisation_name"))
            registration_number = clean_text(
                record.get("organisation_registration_number")
            )

            if not organisation_name:
                skipped += 1
                continue
            #duplicate check
            if registration_number:
                existing_result = await conn.execute(
                    text(
                        """
                        SELECT "OrganisationId"
                        FROM "Organisations"
                        WHERE "OrganisationRegistrationNumber" = :registration_number
                        """
                    ),
                    {"registration_number": registration_number},
                )

                if existing_result.fetchone():
                    skipped += 1
                    continue
            country_id = await lookup_country_id(
                conn,
                record.get("country_code"),
            )

            partner_type_code = await valid_partner_type(
                conn,
                record.get("partner_type_code"),
            )

            category_code = await valid_category(
                conn,
                record.get("category_code"),
            )

            organisation_size_code = await valid_organisation_size(
                conn,
                record.get("organisation_size_code"),
            )

            result = await conn.execute(
                text(

                    """
                    INSERT INTO "Organisations" (
                    "CountryId",
                    "CategoryTypeCode",
                    "PartnerTypeCode",
                    "OrganisationSizeCode",
                    "OrganisationName",
                    "PrimaryOrganisationName",
                    "PartnerTypeAssignmentMethod",
                    "CompanyName",
                    "OrganisationRegistrationNumber",
                    "CityName",
                    "StateName",
                    "WebsiteUrl",
                    "SustainabilityUrl",
                    "PrimaryEmailAddress",
                    "SourceIndustryCodeType",
                    "SourceIndustryCode",
                    "SourceIndustryDescription",
                    "SourceName",
                    "SourceUrl"
                    )
                    VALUES (
                        :country_id,
                        :category_code,
                        :partner_type_code,
                        :organisation_size_code,
                        :organisation_name,
                        :primary_organisation_name,
                        :partner_type_assignment_method,
                        :company_name,
                        :registration_number,
                        :city_name,
                        :state_name,
                        :website_url,
                        :sustainability_url,
                        :primary_email_address,
                        :source_industry_code_type,
                        :source_industry_code,
                        :source_industry_description,
                        :source_name,
                        :source_url
                    )
                    RETURNING "OrganisationId"
                    """
                ),
                {
                    "country_id": country_id,
                    "category_code": category_code,
                    "partner_type_code": partner_type_code,
                    "organisation_size_code": organisation_size_code,
                    "organisation_name": organisation_name,
                    "primary_organisation_name": clean_text(record.get("primary_organisation_name")),
                    "partner_type_assignment_method": clean_text(record.get("partner_type_assignment_method")),
                    "company_name": clean_text(record.get("company_name")),
                    "registration_number": registration_number,
                    "city_name": clean_text(record.get("city_name")),
                    "state_name": clean_text(record.get("state_name")),
                    "website_url": clean_text(record.get("website_url")),
                    "sustainability_url": clean_text(record.get("sustainability_url")),
                    "primary_email_address": clean_text(record.get("primary_email_address")),
                    "source_industry_code_type": clean_text(record.get("source_industry_code_type")),
                    "source_industry_code": clean_text(record.get("source_industry_code")),
                    "source_industry_description": clean_text(record.get("source_industry_description")),
                    "source_name": clean_text(record.get("source")),
                    "source_url": clean_text(record.get("source_url")),
                },
            )

            if result.fetchone():
                inserted += 1

    await engine.dispose()

    return {
        "received": len(records),
        "inserted": inserted,
        "skipped": skipped,
        "status": "successful",
    }