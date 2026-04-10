"""
Example database models for the template repository.
"""

from sqlalchemy import Column, DateTime, Integer, String, func

from src.db.base import Base


class ExampleTable(Base):
    __tablename__ = "example_table"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<ExampleTable id={self.id} name={self.name}>"
