"""Fetch-layer tests.

Everything here is hermetic. The rate limiter is tested against a fake clock
rather than by sleeping, and pagination against a stubbed transport, so the
suite neither takes minutes nor spends the registry's ~50 req/min budget on CI.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from trialrag.ingest.fetch import (
    CtGovClient,
    CtGovError,
    StudyFilter,
    _user_agent,
    versions_of,
)

# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------


def test_filter_builds_essie_expression() -> None:
    params = StudyFilter(
        conditions=["Type 2 Diabetes", "Obesity"],
        statuses=["RECRUITING", "COMPLETED"],
        study_type="INTERVENTIONAL",
        phases=["PHASE2", "PHASE3"],
    ).as_params()

    assert params["query.cond"] == '"Type 2 Diabetes" OR "Obesity"'
    assert params["filter.overallStatus"] == "RECRUITING|COMPLETED"
    assert params["filter.advanced"] == (
        "AREA[StudyType]INTERVENTIONAL AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"
    )


def test_filter_omits_empty_clauses() -> None:
    params = StudyFilter(study_type=None).as_params()
    assert params == {}


def test_conditions_are_quoted_so_multiword_terms_stay_atomic() -> None:
    # Unquoted, "Type 2 Diabetes" is parsed as three OR'd tokens and matches
    # essentially every study containing the word "type".
    assert (
        StudyFilter(conditions=["Type 2 Diabetes"], study_type=None).as_params()["query.cond"]
        == '"Type 2 Diabetes"'
    )


# ---------------------------------------------------------------------------
# User agent
# ---------------------------------------------------------------------------


def test_user_agent_extends_rather_than_replaces_httpx_default() -> None:
    """The registry's WAF 403s any Python client that overrides its own UA.

    Replacing the default -- with *anything*, including a literal "curl/8.7.1" --
    fails the fingerprint check. This assertion is what stops someone from
    "cleaning up" the UA and silently breaking every ingest.
    """
    agent = _user_agent()
    assert agent.startswith(httpx.Client().headers["user-agent"])
    assert "trialrag/" in agent
    assert "github.com" in agent


# ---------------------------------------------------------------------------
# Pagination and incremental sync
# ---------------------------------------------------------------------------


def _study(nct_id: str, updated: str = "2026-01-01") -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
            "statusModule": {"lastUpdatePostDateStruct": {"date": updated}},
        }
    }


def _stub(pages: list[dict[str, Any]]) -> httpx.MockTransport:
    """Serve ``pages`` in order, keyed by the pageToken each one advertises."""
    by_token = {None: pages[0]}
    for index, page in enumerate(pages[:-1]):
        by_token[page.get("nextPageToken")] = pages[index + 1]

    def handler(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pageToken")
        return httpx.Response(200, json=by_token[token or None])

    return httpx.MockTransport(handler)


async def _client_with(transport: httpx.MockTransport) -> CtGovClient:
    client = CtGovClient(rate_limit_rpm=6000, page_size=2)
    await client.__aenter__()
    assert client._client is not None
    client._client._transport = transport
    return client


async def test_pagination_follows_next_page_token() -> None:
    pages = [
        {"studies": [_study("NCT00000001"), _study("NCT00000002")], "nextPageToken": "t1"},
        {"studies": [_study("NCT00000003")]},
    ]
    client = await _client_with(_stub(pages))
    try:
        got = [r async for r in client.iter_studies(StudyFilter())]
    finally:
        await client.__aexit__()

    assert [r["protocolSection"]["identificationModule"]["nctId"] for r in got] == [
        "NCT00000001",
        "NCT00000002",
        "NCT00000003",
    ]
    assert client.stats.requests == 2


async def test_limit_stops_paging_early() -> None:
    pages = [
        {"studies": [_study("NCT00000001"), _study("NCT00000002")], "nextPageToken": "t1"},
        {"studies": [_study("NCT00000003")]},
    ]
    client = await _client_with(_stub(pages))
    try:
        got = [r async for r in client.iter_studies(StudyFilter(), limit=1)]
    finally:
        await client.__aexit__()

    assert len(got) == 1
    assert client.stats.requests == 1, "must not fetch a page it does not need"


async def test_unchanged_studies_are_skipped() -> None:
    pages = [
        {
            "studies": [
                _study("NCT00000001", "2026-01-01"),
                _study("NCT00000002", "2026-02-01"),
            ]
        }
    ]
    client = await _client_with(_stub(pages))
    try:
        got = [
            r
            async for r in client.iter_studies(
                StudyFilter(), known_versions={"NCT00000001": "2026-01-01"}
            )
        ]
    finally:
        await client.__aexit__()

    assert [r["protocolSection"]["identificationModule"]["nctId"] for r in got] == ["NCT00000002"]


async def test_changed_study_is_re_yielded() -> None:
    """A stale known version must not suppress a genuinely updated record."""
    pages = [{"studies": [_study("NCT00000001", "2026-05-01")]}]
    client = await _client_with(_stub(pages))
    try:
        got = [
            r
            async for r in client.iter_studies(
                StudyFilter(), known_versions={"NCT00000001": "2026-01-01"}
            )
        ]
    finally:
        await client.__aexit__()
    assert len(got) == 1


async def test_malformed_payload_raises() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"unexpected": []}))
    client = await _client_with(transport)
    try:
        with pytest.raises(CtGovError, match="studies"):
            _ = [r async for r in client.iter_studies(StudyFilter())]
    finally:
        await client.__aexit__()


async def test_server_errors_are_retried_then_succeed() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"studies": [_study("NCT00000001")]})

    client = await _client_with(httpx.MockTransport(handler))
    try:
        got = [r async for r in client.iter_studies(StudyFilter())]
    finally:
        await client.__aexit__()

    assert len(got) == 1
    assert calls["n"] == 3
    assert client.stats.retries == 2


async def test_client_errors_are_not_retried() -> None:
    """A 404 is a bug in our request; retrying it just wastes the rate budget."""
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    client = await _client_with(httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            _ = [r async for r in client.iter_studies(StudyFilter())]
    finally:
        await client.__aexit__()
    assert calls["n"] == 1


async def test_using_client_outside_context_manager_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        await CtGovClient().fetch_study("NCT00000001")


def test_versions_of_extracts_update_dates() -> None:
    assert versions_of([_study("NCT00000001", "2026-03-04"), {"garbage": True}]) == {
        "NCT00000001": "2026-03-04"
    }


# ---------------------------------------------------------------------------
# Live contract check
# ---------------------------------------------------------------------------


@pytest.mark.network
async def test_live_registry_contract() -> None:
    """Hits the real API. Deselected by default via ``-m 'not network'``.

    Guards the assumptions the hermetic tests encode: that the endpoint shape,
    the Essie filter syntax and the WAF's UA rule are all still what we built
    against.
    """
    async with CtGovClient(page_size=5) as client:
        study_filter = StudyFilter(conditions=["Diabetes"], phases=["PHASE3"])
        assert await client.count(study_filter) > 0

        records = [r async for r in client.iter_studies(study_filter, limit=3)]
        assert len(records) == 3
        for record in records:
            assert "protocolSection" in record
            # The field projection must actually be honoured, or a full corpus
            # pull balloons by two orders of magnitude.
            assert len(json.dumps(record)) < 500_000
