#!/usr/bin/env python3
"""
Medical Guideline Collection Script
Downloads authoritative medical guidelines for RAG system.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DATA_DIR = Path("data/guidelines")

EXT_BY_TYPE = {"pdf": ".pdf", "html": ".html"}  # falls back to .txt

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("guidelines")


@dataclass(frozen=True)
class GuidelineSource:
    key: str
    name: str
    url: str
    type: str  # "pdf" | "html"
    description: str
    author: str
    year: int
    topic: str


GUIDELINE_SOURCES: list[GuidelineSource] = [
    GuidelineSource(
        key="cdc_opioid_2022",
        name="CDC Clinical Practice Guideline for Prescribing Opioids for Pain — United States, 2022",
        url="https://stacks.cdc.gov/view/cdc/122248/cdc_122248_DS1.pdf",
        type="pdf",
        description=(
            "Evidence-based recommendations for clinicians prescribing opioids for "
            "acute, subacute, and chronic pain in outpatients aged \u226518 years."
        ),
        author="Centers for Disease Control and Prevention",
        year=2022,
        topic="pain management",
    ),
    GuidelineSource(
        key="cdc_overdose_prevention",
        name="CDC Guideline Recommendations and Guiding Principles for Overdose Prevention",
        url="https://www.cdc.gov/overdose-prevention/hcp/clinical-guidance/recommendations-and-principles.html",
        type="html",
        description="Clinical guidance and principles for overdose prevention",
        author="Centers for Disease Control and Prevention",
        year=2023,
        topic="overdose prevention",
    ),
    GuidelineSource(
        key="who_essential_medicines_2023",
        name="WHO Model List of Essential Medicines – 23rd List, 2023",
        url="https://www.who.int/publications/i/item/WHO-MHP-HPS-EML-2023.02",
        type="html",
        description="Core list of the most effective and safe medicines needed in a health system",
        author="World Health Organization",
        year=2023,
        topic="essential medicines",
    ),
    GuidelineSource(
        key="cdc_diabetes_2023",
        name="CDC Diabetes Prevention and Control Resources",
        url="https://www.cdc.gov/diabetes/professional-info/index.html",
        type="html",
        description="Evidence-based guidelines and resources for diabetes prevention and management",
        author="Centers for Disease Control and Prevention",
        year=2023,
        topic="diabetes",
    ),
    GuidelineSource(
        key="uspreventive_services",
        name="U.S. Preventive Services Task Force Recommendations",
        url="https://www.uspreventiveservicestaskforce.org/uspstf/recommendation-topics/uspstf-a-and-b-recommendations",
        type="html",
        description="Evidence-based preventive service recommendations for primary care",
        author="U.S. Preventive Services Task Force",
        year=2024,
        topic="preventive care",
    ),
]


def build_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Session with retry/backoff on transient errors, reused across downloads."""
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(session: requests.Session, url: str, filepath: Path) -> bool:
    """Stream a URL to disk. Returns True on success, False on any failure."""
    log.info("Downloading %s", url)
    try:
        with session.get(url, timeout=60, stream=True) as response:
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        log.info("Saved to %s", filepath)
        return True
    except requests.exceptions.HTTPError as e:
        log.error("HTTP %s for %s: %s", e.response.status_code, url, e.response.reason)
    except requests.exceptions.RequestException as e:
        log.error("Request failed for %s: %s", url, e)
    # clean up partial file so a later run doesn't treat it as "already downloaded"
    filepath.unlink(missing_ok=True)
    return False


def file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def collect_one(session: requests.Session, source: GuidelineSource) -> dict | None:
    """Download (if needed) + hash + write metadata for a single source."""
    ext = EXT_BY_TYPE.get(source.type, ".txt")
    filepath = DATA_DIR / f"{source.key}{ext}"

    if filepath.exists():
        log.info("Already downloaded: %s", filepath)
    elif not download_file(session, source.url, filepath):
        return None

    metadata = {
        **asdict(source),
        "id": source.key,
        "source_url": source.url,
        "local_path": str(filepath.absolute()),
        "file_type": source.type,
        "file_size": filepath.stat().st_size,
        "sha256": file_hash(filepath),
    }
    del metadata["key"], metadata["url"], metadata["type"]  # de-dup vs. the renamed keys above

    write_json(DATA_DIR / f"{source.key}_metadata.json", metadata)
    log.info("Size: %s bytes | SHA256: %s...", f"{metadata['file_size']:,}", metadata["sha256"][:16])
    return metadata


def collect_guidelines() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Starting medical guidelines collection -> %s", DATA_DIR.absolute())

    session = build_session()
    collected: list[dict] = []
    failed: list[str] = []

    for source in GUIDELINE_SOURCES:
        log.info("Processing: %s", source.name)
        result = collect_one(session, source)
        (collected if result else failed).append(result or source.key)

    log.info("=" * 50)
    log.info("Collection complete: %d succeeded, %d failed", len(collected), len(failed))
    if failed:
        log.warning("Failed items: %s", ", ".join(failed))

    write_json(
        DATA_DIR / "collection_manifest.json",
        {"total_collected": len(collected), "total_failed": len(failed), "guidelines": collected},
    )
    log.info("Manifest saved to %s", DATA_DIR / "collection_manifest.json")
    return collected


if __name__ == "__main__":
    collect_guidelines()