# Zoho Integration Module

This directory contains the sample code framework to connect the PostgreSQL database to the RNB Media Zoho CRM instance.

## Key Components
**integration_template.py**: A functional baseline for OAuth 2.0 authentication and lead ingestion.

## Deployment Requirements
**Environment Variables**: Sensitive credentials (**Client ID**, **Client Secret**, **Refresh Token**) must **not** be hardcoded. Use a secure `.env` file.
**Layout Mapping**: Ensure the Zoho "Group 7 Testing" layout is configured with all 15 required fields.
