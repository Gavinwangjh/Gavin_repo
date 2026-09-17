"""initial migration v2

Revision ID: 38c5fa2a20af
Revises: 909cf6558bf1
Create Date: 2026-04-10 20:02:57.868717

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "38c5fa2a20af"
down_revision: str | Sequence[str] | None = "909cf6558bf1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ETL lookup and organisation tables."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "Countries" (
            "Id" INTEGER PRIMARY KEY,
            "CountryCode" INTEGER NOT NULL UNIQUE,
            "CountryName" TEXT NOT NULL UNIQUE
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "PartnerType" (
            "PartnerTypeCode" INTEGER PRIMARY KEY,
            "PartnerTypeCodeDescription" TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "OrganisationSize" (
            "OrganisationSizeCode" INTEGER PRIMARY KEY,
            "OrganisationSizeDescription" TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "CategoryTypes" (
            "CategoryCode" INTEGER PRIMARY KEY,
            "CategoryDescription" TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "Organisations" (
            "OrganisationId" SERIAL PRIMARY KEY,
            "CountryId" INTEGER REFERENCES "Countries"("Id"),
            "CategoryTypeCode" INTEGER REFERENCES "CategoryTypes"("CategoryCode"),
            "PartnerTypeCode" INTEGER REFERENCES "PartnerType"("PartnerTypeCode"),
            "OrganisationSizeCode" INTEGER REFERENCES "OrganisationSize"("OrganisationSizeCode"),
            "OrganisationName" TEXT,
            "PrimaryOrganisationName" TEXT,
            "PartnerTypeAssignmentMethod" TEXT,
            "CompanyName" TEXT,
            "OrganisationRegistrationNumber" TEXT,
            "CityName" TEXT,
            "StateName" TEXT,
            "WebsiteUrl" TEXT,
            "SustainabilityUrl" TEXT,
            "PrimaryEmailAddress" TEXT,
            "SourceIndustryCodeType" TEXT,
            "SourceIndustryCode" TEXT,
            "SourceIndustryDescription" TEXT,
            "SourceName" TEXT,
            "SourceUrl" TEXT,
            "CreatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    op.execute(
        """
        INSERT INTO "Countries" ("Id", "CountryCode", "CountryName") VALUES
            (61, 61, 'Australia'),
            (44, 44, 'United Kingdom'),
            (1, 1, 'United States')
        ON CONFLICT ("Id") DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO "PartnerType" ("PartnerTypeCode", "PartnerTypeCodeDescription") VALUES
            (1, 'NGO'),
            (2, 'Brand'),
            (3, 'CSO'),
            (4, 'Community'),
            (5, 'Consumer'),
            (6, 'Expert')
        ON CONFLICT ("PartnerTypeCode") DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO "OrganisationSize" ("OrganisationSizeCode", "OrganisationSizeDescription") VALUES
            (1, 'Small'),
            (2, 'Medium'),
            (3, 'Large')
        ON CONFLICT ("OrganisationSizeCode") DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO "CategoryTypes" ("CategoryCode", "CategoryDescription") VALUES
            (1, 'Agriculture, Forestry and Fishing'),
            (2, 'Mining'),
            (3, 'Manufacturing'),
            (4, 'Electricity, Gas, Water and Waste Services'),
            (5, 'Construction'),
            (6, 'Wholesale Trade'),
            (7, 'Retail Trade'),
            (8, 'Accommodation and Food Services'),
            (9, 'Transport, Postal and Warehousing'),
            (10, 'Information Media and Telecommunications'),
            (11, 'Financial and Insurance Services'),
            (12, 'Rental, Hiring and Real Estate Services'),
            (13, 'Professional, Scientific and Technical Services'),
            (14, 'Administrative and Support Services'),
            (15, 'Public Administration and Safety'),
            (16, 'Education and Training'),
            (17, 'Health Care and Social Assistance'),
            (18, 'Arts and Recreation Services'),
            (19, 'Other Services')
        ON CONFLICT ("CategoryCode") DO NOTHING
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS "idx_organisations_registration"
        ON "Organisations" ("OrganisationRegistrationNumber")
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS "idx_organisations_country_registration"
        ON "Organisations" ("CountryId", "OrganisationRegistrationNumber")
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS "idx_organisations_source"
        ON "Organisations" ("SourceName")
        """
    )


def downgrade() -> None:
    """Drop the ETL tables."""
    op.execute('DROP INDEX IF EXISTS "idx_organisations_source"')
    op.execute('DROP INDEX IF EXISTS "idx_organisations_country_registration"')
    op.execute('DROP INDEX IF EXISTS "idx_organisations_registration"')
    op.execute('DROP TABLE IF EXISTS "Organisations"')
    op.execute('DROP TABLE IF EXISTS "CategoryTypes"')
    op.execute('DROP TABLE IF EXISTS "OrganisationSize"')
    op.execute('DROP TABLE IF EXISTS "PartnerType"')
    op.execute('DROP TABLE IF EXISTS "Countries"')
