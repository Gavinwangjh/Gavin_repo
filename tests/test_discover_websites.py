from src.temporal.activities.discover_websites import (
    build_candidate_urls,
    build_domain_base,
    discover_website_for_org,
    page_matches_organisation,
    select_tlds,
)


class DummySession:
    def get(self, *args, **kwargs):
        raise AssertionError("No network request should be made")


def test_build_domain_base_removes_company_suffixes_and_symbols():
    assert build_domain_base("Example Holdings Pty Ltd.") == "example"


def test_select_tlds_supports_country_aliases():
    assert select_tlds("AU")[0] == ".com.au"
    assert select_tlds("GB")[0] == ".co.uk"
    assert select_tlds(None) == (".com", ".org")


def test_build_candidate_urls_includes_www_and_apex_domains():
    assert build_candidate_urls("example", "AU")[:4] == [
        "https://www.example.com.au",
        "https://example.com.au",
        "https://www.example.org.au",
        "https://example.org.au",
    ]


def test_page_matches_cleaned_organisation_name():
    assert page_matches_organisation(
        html="Example delivers community programs across Australia.",
        title="Example",
        organisation_name="Example Pty Ltd",
        domain_base="example",
    )


def test_discover_website_for_org_preserves_existing_website():
    org = {
        "organisation_name": "Example Inc",
        "website": "example.com",
    }

    assert discover_website_for_org(org, DummySession()) == "https://example.com"
    assert org["website_lookup_status"] == "website_already_available"
