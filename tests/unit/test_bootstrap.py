"""Regression test for the SSL trust-store bootstrap fix.

Without this, some macOS Python builds fail every Voyage API call with
``SSLCertVerificationError: unable to get local issuer certificate`` --
``aiohttp`` (used internally by the Voyage SDK) cannot find a usable trust
store on its own, while ``httpx`` (ClinicalTrials.gov) is unaffected, which
made the failure look Voyage-specific rather than environmental. Verified live
against the real API: the same key failed via ``voyageai.AsyncClient`` and
succeeded via ``curl`` until this fix was applied.
"""

from __future__ import annotations

import importlib
import os

import pytest

from trialrag import bootstrap


@pytest.fixture(autouse=True)
def _clean_ssl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def test_sets_ssl_cert_file_from_certifi_when_unset() -> None:
    bootstrap.ensure_ssl_trust_store()
    assert os.environ.get("SSL_CERT_FILE", "").endswith("cacert.pem")


def test_does_not_override_an_existing_ssl_cert_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/some/custom/bundle.pem")
    bootstrap.ensure_ssl_trust_store()
    assert os.environ["SSL_CERT_FILE"] == "/some/custom/bundle.pem"


def test_does_not_override_an_existing_ssl_cert_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_DIR", "/some/custom/dir")
    bootstrap.ensure_ssl_trust_store()
    assert "SSL_CERT_FILE" not in os.environ


def test_runs_as_an_import_side_effect() -> None:
    """The whole point is that entry points need only `import trialrag.bootstrap`
    -- if this regresses to call-only, cli.py's import-based wiring goes silent."""
    importlib.reload(bootstrap)
    assert os.environ.get("SSL_CERT_FILE", "").endswith("cacert.pem")
