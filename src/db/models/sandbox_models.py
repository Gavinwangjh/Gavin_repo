from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.db.base import Base


class Organisations(Base):
    __tablename__ = "Organisations"

    # PK
    OrganisationId = Column(Integer, primary_key=True, unique=True, autoincrement=True)

    # FKs
    CountryId = Column(Integer, ForeignKey("Countries.Id"))
    CategoryTypeCode = Column(Integer, ForeignKey("CategoryTypes.CategoryCode"))
    PartnerTypeCode = Column(Integer, ForeignKey("PartnerType.PartnerTypeCode"))
    PartyIdentifier = Column(Integer, ForeignKey("Party.PartyIdentifier"))
    OrganisationSizeCode = Column(Integer, ForeignKey("OrganisationSize.OrganisationSizeCode"))

    # Other fields
    OrganisationName = Column(String(1000), nullable=False)
    OrganisationRegistrationNumber = Column(String(50))
    RegionName = Column(String(500))
    AddressLine1Description = Column(String(1000))
    CityName = Column(String(500))
    StateName = Column(String(500))
    WebsiteUrl = Column(String(500))
    SustainabilityUrl = Column(String(500))
    PrimaryEmailAddress = Column(String(255))

    country = relationship("Countries", back_populates="organisation")
    category_code = relationship(
        "CategoryTypes", back_populates="organisation", foreign_keys=[CategoryTypeCode, PartnerTypeCode]
    )
    party = relationship("Party", back_populates="organisation")
    partner_type = relationship("PartnerType", back_populates="organisation")
    organisation_size = relationship("OrganisationSize", back_populates="organisation")


class Countries(Base):
    __tablename__ = "Countries"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    CountryCode = Column(Integer, unique=True, nullable=False)
    CountryName = Column(String(255), unique=True, nullable=False)
    Alpha2 = Column(String(2), unique=True, nullable=False)
    Alpha3 = Column(String(3), unique=True, nullable=False)

    # Relationships
    organisation = relationship("Organisations", back_populates="country")

    def __repr__(self):
        return (
            f"<Countries | CountryId={self.CountryId}, "
            f"CountryCode={self.CountryCode}, "
            f"CountryName={self.CountryName},"
            f" Alpha2={self.Alpha2}, Alpha3={self.Alpha3}>"
        )


class Party(Base):
    __tablename__ = "Party"

    PartyIdentifier = Column(Integer, primary_key=True)

    organisation = relationship("Organisations", back_populates="party")

    def __repr__(self):
        return f"<Party | PartyIdentifier={self.PartyIdentifier}>"


class CategoryTypes(Base):
    __tablename__ = "CategoryTypes"

    CategoryCode = Column(Integer, primary_key=True)
    PartnerTypeCode = Column(Integer, ForeignKey("PartnerType.PartnerTypeCode"))
    CategoryDescription = Column(String(1000), unique=True, nullable=False)

    partner_type = relationship("PartnerType", back_populates="category_code")
    organisation = relationship("Organisations", back_populates="category_code")

    def __repr__(self):
        return (
            f"<CategoryTypes | CategoryCode={self.CategoryCode}, Description={self.CategoryDescription}, "
            f"PartnerTypeCode={self.PartnerTypeCode}>"
        )


class OrganisationSize(Base):
    __tablename__ = "OrganisationSize"

    OrganisationSizeCode = Column(Integer, primary_key=True)
    OrganisationSizeDescription = Column(String(255), unique=True)

    organisation = relationship("Organisations", back_populates="organisation_size")

    def __repr__(self):
        return (
            f"<OrganisationSize | OrganisationSizeCode={self.OrganisationSizeCode}, "
            f"OrganisationSizeDescription={self.OrganisationSizeDescription}>"
        )


class PartnerType(Base):
    __tablename__ = "PartnerType"

    PartnerTypeCode = Column(Integer, primary_key=True)
    PartnerTypeCodeDescription = Column(String(255), nullable=False)

    category_code = relationship("CategoryTypes", back_populates="partner_type")

    def __repr__(self):
        return f"<PartnerDetails | PartnerTypeCode={self.PartnerTypeCode}>"
