"""Offline corpus pipeline: fetch -> parse -> chunk -> embed -> load.

Each stage is independently runnable and idempotent. Raw API payloads are
archived before parsing, so re-chunking or re-embedding the corpus never costs
another request against ClinicalTrials.gov's ~50 req/min budget.
"""
