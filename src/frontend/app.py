from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ETL_SCRIPT = PROJECT_ROOT / "src" / "temporal" / "scripts" / "run_etl.py"
LOG_DIR = PROJECT_ROOT / "logs"

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("Database configuration is missing. Expected DATABASE_URL.")
TEMPORAL_UI_URL = os.getenv("TEMPORAL_UI_URL", "http://localhost:8080")

WORKING_SOURCES = {
    "Australia - data.gov.au": "au",
    "Australia - WGEA": "au_wgea",
    "United Kingdom - Companies House": "uk",
}

ORGANISATIONS_QUERY = text(
    """
    SELECT
      o."OrganisationId" AS "Organisation ID",
      c."CountryName" AS "Country",
      o."PrimaryOrganisationName" AS "Primary Organisation",
      o."CompanyName" AS "Company Name",
      o."OrganisationName" AS "Organisation Name",
      o."OrganisationRegistrationNumber" AS "Registration Number",
      pt."PartnerTypeCodeDescription" AS "Partner Type",
      o."PartnerTypeAssignmentMethod" AS "Partner Type Method",
      o."CategoryTypeCode" AS "Category Code",
      ct."CategoryDescription" AS "Category",
      os."OrganisationSizeDescription" AS "Organisation Size",
      o."WebsiteUrl" AS "Website",
      o."SustainabilityUrl" AS "Sustainability URL",
      o."PrimaryEmailAddress" AS "Email",
      o."SourceIndustryCodeType" AS "Industry Code Type",
      o."SourceIndustryCode" AS "Industry Code",
      o."SourceIndustryDescription" AS "Industry Description",
      o."SourceName" AS "Source"
    FROM "Organisations" o
    LEFT JOIN "Countries" c
      ON o."CountryId" = c."Id"
    LEFT JOIN "PartnerType" pt
      ON o."PartnerTypeCode" = pt."PartnerTypeCode"
    LEFT JOIN "CategoryTypes" ct
      ON o."CategoryTypeCode" = ct."CategoryCode"
    LEFT JOIN "OrganisationSize" os
      ON o."OrganisationSizeCode" = os."OrganisationSizeCode"
    ORDER BY o."OrganisationId" ASC
    """
)


async def _load_organisations() -> list[dict[str, Any]]:
    engine = create_async_engine(DATABASE_URL)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(ORGANISATIONS_QUERY)
            return [dict(row) for row in result.mappings().all()]
    finally:
        await engine.dispose()


def load_organisations() -> pd.DataFrame:
    rows = asyncio.run(_load_organisations())
    return pd.DataFrame(rows)


def trigger_workflow(source: str, limit: int) -> tuple[int, Path]:
    if not RUN_ETL_SCRIPT.exists():
        raise FileNotFoundError(f"Workflow trigger script not found: {RUN_ETL_SCRIPT}")

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"frontend_etl_{source}_{timestamp}.log"

    command = [
        sys.executable,
        str(RUN_ETL_SCRIPT),
        "--source",
        source,
        "--limit",
        str(limit),
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return process.pid, log_path


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    left, middle, right = st.columns(3)

    with left:
        country_options = sorted(df["Country"].dropna().unique().tolist())
        selected_countries = st.multiselect("Country", country_options)

    with middle:
        source_options = sorted(df["Source"].dropna().unique().tolist())
        selected_sources = st.multiselect("Source", source_options)

    with right:
        category_options = sorted(df["Category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Category", category_options)

    search_text = st.text_input("Search organisation name or registration number")

    filtered = df.copy()

    if selected_countries:
        filtered = filtered[filtered["Country"].isin(selected_countries)]
    if selected_sources:
        filtered = filtered[filtered["Source"].isin(selected_sources)]
    if selected_categories:
        filtered = filtered[filtered["Category"].isin(selected_categories)]
    if search_text:
        search_text = search_text.strip().lower()
        mask = (
            filtered["Organisation Name"].fillna("").str.lower().str.contains(search_text)
            | filtered["Company Name"].fillna("").str.lower().str.contains(search_text)
            | filtered["Registration Number"].fillna("").astype(str).str.lower().str.contains(search_text)
        )
        filtered = filtered[mask]

    return filtered


def render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    with_website = int(df["Website"].notna().sum()) if "Website" in df else 0
    with_sustainability = int(df["Sustainability URL"].notna().sum()) if "Sustainability URL" in df else 0
    with_category = int(df["Category"].notna().sum()) if "Category" in df else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", total)
    col2.metric("With website", with_website)
    col3.metric("With sustainability URL", with_sustainability)
    col4.metric("With category", with_category)


def render_trigger_tab() -> None:
    st.subheader("Trigger ETL workflow")

    with st.form("trigger_workflow_form"):
        source_label = st.selectbox("Source", list(WORKING_SOURCES.keys()))
        limit = st.number_input("Extraction limit", min_value=1, max_value=5000, value=25, step=1)
        submitted = st.form_submit_button("Start workflow", type="primary")

    if submitted:
        source = WORKING_SOURCES[source_label]
        try:
            pid, log_path = trigger_workflow(source, int(limit))
            st.success(f"Workflow trigger started for source '{source}' with limit {limit}.")
            st.write(f"Process ID: `{pid}`")
            st.write(f"Log file: `{log_path.relative_to(PROJECT_ROOT)}`")
            st.link_button("Open Temporal UI", TEMPORAL_UI_URL)
        except Exception as exc:
            st.error(f"Could not start workflow: {exc}")


def render_records_tab() -> None:
    st.subheader("Loaded organisation records")

    if st.button("Refresh records"):
        st.rerun()

    try:
        df = load_organisations()
    except Exception as exc:
        st.error(f"Could not load organisation records: {exc}")
        return

    if df.empty:
        st.warning("No organisation records are currently loaded.")
        return

    filtered_df = filter_dataframe(df)
    render_metrics(filtered_df)

    display_columns = [
        "Organisation ID",
        "Country",
        "Organisation Name",
        "Registration Number",
        "Category Code",
        "Category",
        "Partner Type",
        "Organisation Size",
        "Industry Code Type",
        "Industry Code",
        "Industry Description",
        "Website",
        "Sustainability URL",
        "Email",
    ]

    available_columns = [column for column in display_columns if column in filtered_df.columns]

    st.dataframe(
        filtered_df[available_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "Category Code": st.column_config.NumberColumn(
                "Category Type Code",
                format="%d",
            ),
            "Website": st.column_config.LinkColumn("Website"),
            "Sustainability URL": st.column_config.LinkColumn("Sustainability URL"),
            "Email": st.column_config.TextColumn("Email"),
        },
    )


st.set_page_config(page_title="Organisation ETL Prototype", layout="wide")
st.title("Organisation ETL Prototype")
st.caption("View loaded organisations and trigger supported ETL workflows.")

records_tab, trigger_tab = st.tabs(["Organisation records", "Trigger workflow"])

with records_tab:
    render_records_tab()

with trigger_tab:
    render_trigger_tab()
