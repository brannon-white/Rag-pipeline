"""Process-startup fixes that must run before any network client is built.

Currently one: pointing OpenSSL at certifi's CA bundle. On some macOS Python
builds (observed: the python.org 3.13 framework build), the process-default SSL
context cannot locate a local trust store, so ``aiohttp`` (used internally by
the Voyage SDK) fails every HTTPS request with
``SSLCertVerificationError: unable to get local issuer certificate`` --
while ``httpx`` (used for ClinicalTrials.gov) and plain ``curl`` are unaffected,
because they resolve certs through a different path and made this failure
confusing to diagnose: the same key, against a different library, "just worked".

Call :func:`ensure_ssl_trust_store` once, as early as possible in every process
entry point (``cli.py``; the API app when it exists), before constructing any
HTTP client.
"""

from __future__ import annotations

import os


def ensure_ssl_trust_store() -> None:
    """Point ``SSL_CERT_FILE``/``SSL_CERT_DIR`` at certifi's bundle if unset.

    A no-op wherever the system trust store already resolves -- it only fills
    in the environment variables Python's ``ssl`` module consults when they are
    still empty, so it cannot override a deliberately configured trust store.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()


# Runs on import, not just on call: every entry point (cli.py; the API app)
# does `from trialrag import bootstrap` as its first trialrag import, so the
# fix lands before any other trialrag module has a chance to construct an
# HTTP client -- without needing a separate function call at each call site,
# which would otherwise force those imports below it out of normal import
# order (E402) for no benefit.
ensure_ssl_trust_store()
