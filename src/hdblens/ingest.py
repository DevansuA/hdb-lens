"""Live data ingestion from data.gov.sg.

Downloads the full HDB resale transactions dataset (Jan-2017 onwards)
via the official initiate-download / poll-download API, so the project
always reflects the latest monthly release — no stale bundled CSVs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from hdblens.config import API_BASE, DATASET_ID, RAW_CSV

logger = logging.getLogger(__name__)

POLL_ATTEMPTS = 20
POLL_INTERVAL_S = 2.0


def _poll_download_url(session: requests.Session, dataset_id: str) -> str:
    """Trigger a dataset export and poll until a presigned URL is ready."""
    session.get(f"{API_BASE}/{dataset_id}/initiate-download", timeout=30).raise_for_status()

    for attempt in range(POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_S)
        resp = session.get(f"{API_BASE}/{dataset_id}/poll-download", timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        if url := data.get("url"):
            logger.info("Download URL ready after %d poll(s)", attempt + 1)
            return url
    raise TimeoutError(f"Dataset {dataset_id} export did not become ready in time.")


def download_resale_data(dest: Path = RAW_CSV, force: bool = False) -> Path:
    """Download the resale transactions CSV. Skips if a fresh copy exists.

    Parameters
    ----------
    dest:  Output path for the CSV.
    force: Re-download even if the file already exists.
    """
    if dest.exists() and not force:
        logger.info("Using cached dataset at %s (pass force=True to refresh)", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        url = _poll_download_url(session, DATASET_ID)
        with session.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_resale_data(force=True)
