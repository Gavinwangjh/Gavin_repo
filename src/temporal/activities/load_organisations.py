import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from temporalio import activity


@activity.defn
async def load_organisations(transformed_data: dict) -> dict:
    records = transformed_data.get("transformed_sample", [])

    if not records:
        return {"inserted": 0, "status": "no_data"}

    database_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(database_url)

    inserted = 0

    def safe_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    async with engine.begin() as conn:
        for r in records:
            org_raw = r.get("organisation_registration_number")
            try:
                org_id = int(str(org_raw).strip())
            except (TypeError, ValueError):
                continue

            country_code = r.get("country_code")
            if country_code:
                country_code = country_code.strip().upper()

            partner_type = safe_int(r.get("partner_type_code"))
            category = safe_int(r.get("category_code"))
            size = safe_int(r.get("organisation_size_code"))

            if partner_type is not None:
                result = await conn.execute(
                    text('SELECT 1 FROM "PartnerType" WHERE "PartnerTypeCode" = :code'),
                    {"code": partner_type},
                )

                if result.fetchone() is None:
                    partner_type = None

            if category is not None:
                print("deb:", r.get("country_code"))
                result = await conn.execute(
                    text('SELECT 1 FROM "CategoryTypes" WHERE "CategoryCode" = :code'), {"code": category}
                )
                # if not result.fetchone():
                if result.fetchone() is None:
                    category = None

            if size is not None:
                result = await conn.execute(
                    text('SELECT 1 FROM "OrganisationSize" WHERE "OrganisationSizeCode" = :code'),
                    {"code": size},
                )

                if result.fetchone() is None:
                    size = None

            result = await conn.execute(
                text("""
                              INSERT INTO "Organisations"(
                                   "OrganisationId",
                                   "OrganisationName",
                                   "CountryCode",
                                   "WebsiteUrl",
                                   "PrimaryEmailAddress",
                                    "CityName",
                                    "PartnerTypeCode",
                                   "CategoryTypeCode",
                                   "OrganisationSizeCode",
                                   "PartyIdentifier"
                                   )

                                   VALUES(
                                   :org_id,
                                   :name,
                                   :country,
                                   :website,
                                   :email,
                                   :city,
                                   :partner_type,
                                   :category,
                                   :size,
                                   :party_id

                                   )
                                   ON CONFLICT ("OrganisationId") DO NOTHING
                                   RETURNING "OrganisationId"
                                   """),
                {
                    "org_id": org_id,
                    "name": r.get("organisation_name"),
                    "country": country_code,
                    "website": r.get("website_url"),
                    "email": r.get("primary_email_address"),
                    "city": r.get("city_name"),
                    "partner_type": partner_type,
                    "category": category,
                    "size": size,
                    "party_id": None,
                },
            )

        row = result.fetchone()
        if row is not None:
            inserted += 1

    return {"inserted": inserted, "status": "successful!"}
