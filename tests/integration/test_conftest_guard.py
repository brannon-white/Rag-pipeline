"""Regression test for the dev-database guard in tests/integration/conftest.py.

A pure string check with no database dependency, so it runs unconditionally --
even with no Postgres reachable at all -- which matters here specifically: the
incident it guards against (the schema-dropping ``db`` fixture pointed at the
dev/ingest database, silently destroying ~30 real API calls' worth of an
in-progress corpus ingest) must never depend on having a database available to
reproduce or verify against.

Imported as ``from integration.conftest import ...`` rather than
``tests.integration.conftest``: this repo has no ``tests/__init__.py``, so
pytest's default import mode treats ``tests/`` as the sys.path root and
``integration`` (which does have an ``__init__.py``) as the top-level package
name -- a dotted ``tests.integration.conftest`` import raises
``ModuleNotFoundError: No module named 'tests'``, and a bare ``from conftest
import ...`` resolves to the *wrong* conftest.py (the repo-root one, found via
pytest's own rootdir conftest discovery, not this directory's).
"""

from __future__ import annotations

import pytest

from integration.conftest import _assert_not_the_dev_database


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://trialrag:trialrag@localhost:5432/trialrag",
        "postgresql://trialrag:trialrag@localhost:5432/trialrag/",
        "postgresql://user:pass@some-host:5432/trialrag",
    ],
)
def test_refuses_the_dev_database_by_name(dsn: str) -> None:
    with pytest.raises(RuntimeError, match="dev/ingest database"):
        _assert_not_the_dev_database(dsn)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://trialrag:trialrag@localhost:5432/trialrag_test",
        "postgresql://user:pass@some-host:5432/trialrag_ci",
    ],
)
def test_allows_a_dedicated_test_database(dsn: str) -> None:
    _assert_not_the_dev_database(dsn)  # must not raise
