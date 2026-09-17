import os

import requests

# Configuration details: These must be loaded from environment variables
# to ensure security and prevent hardcoding of sensitive credentials.
CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")

# The API endpoint for the Leads module in Zoho CRM
ZOHO_API_URL = "https://www.zohoapis.com/crm/v2/Leads"


def get_access_token():
    """Exchanges the permanent refresh_token for a temporary access_token."""
    url = (
        f"https://accounts.zoho.com/oauth/v2/token?"
        f"refresh_token={REFRESH_TOKEN}"
        f"&client_id={CLIENT_ID}"
        f"&client_secret={CLIENT_SECRET}"
        f"&grant_type=refresh_token"
    )
    response = requests.post(url)
    return response.json().get("access_token")


def push_to_zoho(data):
    """Pushes a record to the Zoho Leads module."""
    access_token = get_access_token()
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}", "Content-Type": "application/json"}

    # Payload structured for the 'Group 7 Testing' layout
    payload = {"data": [data], "trigger": ["approval", "workflow", "blueprint"]}

    response = requests.post(ZOHO_API_URL, json=payload, headers=headers)
    return response.json()


# Example mapping structure to be populated by the team
sample_lead = {
    "OrganisationName": "Example Corp",
    "OrganisationRegistrationNumber": "ABN123456789",
    "CityName": "Adelaide",
    "PrimaryEmailAddress": "contact@example.com",
}
