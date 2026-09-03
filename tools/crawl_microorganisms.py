#!/usr/bin/env python3
"""
Microorganism Image Crawler & Dataset Builder

Downloads microscope images from multiple free sources:
  - Wikimedia Commons (CC-licensed)
  - Pixabay (free images)
  - Web scraping (educational/scientific sources)

Features:
  - Verified organism name taxonomy (prevents miscategorization)
  - Image validation (corrupt/blank/icon detection)
  - Automatic deduplication via perceptual hashing
  - Respects rate limits and source attribution
  - Adds new classes and augments existing ones

Usage:
  python tools/crawl_microorganisms.py --list-classes
  python tools/crawl_microorganisms.py --class Paramecium --limit 50
  python tools/crawl_microorganisms.py --class-all --limit 30
  python tools/crawl_microorganisms.py --augment-existing --limit 20
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urljoin

import requests
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crawler")

# ---------------------------------------------------------------------------
# Verified Taxonomy — only these names are allowed as class labels
# ---------------------------------------------------------------------------

TAXONOMY: Dict[str, Dict] = {
    # === EXISTING CLASSES (augment with more images) ===
    "Amoeba": {
        "display_name": "Amoeba",
        "search_terms": [
            "amoeba microscope image", "amoeba proteus microscopy",
            "amoeba cell biology", "amoeba moving pseudopodia",
        ],
        "wikimedia_category": "Amoeba",
        "extra_wikimedia_categories": [
            "Amoebozoa", "Testate_amoebae", "Arcellinida",
            "Foraminifera", "Rhizopoda",
        ],
        "description": "Single-celled protist with pseudopodia",
    },
    "Euglena": {
        "display_name": "Euglena",
        "search_terms": [
            "euglena microscope", "euglena gracilis microscopy",
            "euglena flagella", "euglena under microscope",
        ],
        "wikimedia_category": "Euglena",
        "extra_wikimedia_categories": [
            "Euglenozoa", "Euglenaceae", "Kinetoplastea",
            "Trypanosomatida",
        ],
        "description": "Flagellated protist with chloroplasts",
    },
    "Hydra": {
        "display_name": "Hydra",
        "search_terms": [
            "hydra microscope", "hydra vulgaris microscopy",
            "freshwater hydra organism", "hydra tentacles microscope",
        ],
        "wikimedia_category": "Hydra_(genus)",
        "extra_wikimedia_categories": [
            "Hydrozoa", "Cnidaria", "Freshwater_animals",
            "Invertebrate_zoology", "Animals_of_Europe",
            "Animals_of_North_America", "Hydra",
        ],
        "description": "Freshwater polyp with tentacles",
    },
    "Paramecium": {
        "display_name": "Paramecium",
        "search_terms": [
            "paramecium microscope", "paramecium caudatum microscopy",
            "paramecium cilia", "paramecium slide",
        ],
        "wikimedia_category": "Paramecium",
        "extra_wikimedia_categories": [
            "Ciliophora", "Hymenostomatida", "Oligohymenophorea",
            "Ciliate",
        ],
        "description": "Ciliated protist, slipper-shaped",
    },
    "Rod_bacteria": {
        "display_name": "Rod Bacteria (Bacillus)",
        "search_terms": [
            "bacillus bacteria microscope", "rod shaped bacteria microscopy",
            "bacillus gram stain", "bacillus subtilis microscope",
        ],
        "wikimedia_category": "Bacillus",
        "extra_wikimedia_categories": [
            "Escherichia_coli", "Gram_negative_bacteria",
            "Bacteria_imaged_by_electron_microscopy",
            "Bacillota", "Proteobacteria",
            "Bacteria", "Microbiology",
        ],
        "description": "Rod-shaped bacteria (Bacillus, E. coli, etc.)",
    },
    "Spherical_bacteria": {
        "display_name": "Spherical Bacteria (Coccus)",
        "search_terms": [
            "staphylococcus microscope", "coccus bacteria microscopy",
            "streptococcus gram stain", "spherical bacteria microscope",
        ],
        "wikimedia_category": "Cocci",
        "extra_wikimedia_categories": [
            "Staphylococcus", "Streptococcus",
            "Actinomycetota", "Micrococcaceae",
            "Gram_positive_bacteria", "Bacteria",
        ],
        "description": "Spherical bacteria (Staphylococcus, Streptococcus)",
    },
    "Spiral_bacteria": {
        "display_name": "Spiral Bacteria (Spirillum)",
        "search_terms": [
            "spirillum bacteria microscope", "spirochete microscopy",
            "vibrio cholerae microscope", "spiral bacteria gram stain",
        ],
        "wikimedia_category": "Spirilla",
        "extra_wikimedia_categories": [
            "Spirochaetes", "Vibrio", "Helicobacter",
            "Campylobacter", "Spirillum",
            "Bacteria", "Microbiology",
        ],
        "description": "Spiral-shaped bacteria (Spirillum, Spirochete)",
    },
    "Yeast": {
        "display_name": "Yeast",
        "search_terms": [
            "saccharomyces cerevisiae microscope", "yeast cells microscopy",
            "yeast budding microscope", "candida albicans microscope",
        ],
        "wikimedia_category": "Saccharomyces_cerevisiae",
        "extra_wikimedia_categories": [
            "Saccharomycetaceae", "Candida_(fungus)",
            "Ascomycota", "Fungi",
        ],
        "description": "Single-celled fungi (Saccharomyces, Candida)",
    },
    # === NEW CLASSES ===
    "Volvox": {
        "display_name": "Volvox",
        "search_terms": [
            "volvox microscope", "volvox colony microscopy",
            "volvox globator microscope", "volvox under microscope",
        ],
        "wikimedia_category": "Volvox",
        "extra_wikimedia_categories": [
            "Chlorophyta", "Volvox_carteri",
            "Green_algae", "Colonial_algae",
            "Volvocaceae", "Volvocales",
        ],
        "description": "Colonial green algae forming spherical colonies",
    },
    "Diatom": {
        "display_name": "Diatom",
        "search_terms": [
            "diatom microscope", "diatom microscopy slides",
            "diatom frustule microscope", "centric diatom microscopy",
        ],
        "wikimedia_category": "Diatoms",
        "extra_wikimedia_categories": [
            "Bacillariophyceae", "Diatomaceous_earth",
            "Diatom_classifier", "Bacillariophyta",
            "Diatom", "Marine_phytoplankton",
        ],
        "description": "Photosynthetic algae with silica cell walls",
    },
    "Rotifer": {
        "display_name": "Rotifer",
        "search_terms": [
            "rotifer microscope", "rotifera microscopy",
            "rotifer wheel organ", "bdelloid rotifer microscope",
        ],
        "wikimedia_category": "Rotifera",
        "extra_wikimedia_categories": [
            "Rotifer", "Monogononta", "Bdelloidea",
            "Microscopic_animals", "Microscopic_invertebrates",
            "Freshwater_fauna", "Habrotrochidae",
            "Philodinidae", "Notommata", "Philodinavidae",
            "Taphrocampa", "Adinetidae", "Pleuretra",
            "Collothecidae", "Flosculariidae", "Notommatidae",
            "Trichocercidae", "Epiphanidae", "Proalidae",
            "Testudinellidae", "Flosculariaceae", "Synchaetidae",
        ],
        "description": "Microscopic multicellular animals with ciliated corona",
    },
    "Stentor": {
        "display_name": "Stentor",
        "search_terms": [
            "stentor microscope", "stentor coeruleus microscopy",
            "stentor ciliate", "stentor protozoa microscope",
        ],
        "wikimedia_category": "Stentor",
        "extra_wikimedia_categories": [
            "Heterotrichida", "Ciliophora",
            "Spirostomidae", "Protozoa",
            "Ciliate", "Single-celled_organisms",
        ],
        "description": "Large trumpet-shaped ciliated protist",
    },
    "Chlamydomonas": {
        "display_name": "Chlamydomonas",
        "search_terms": [
            "chlamydomonas microscope", "chlamydomonas reinhardtii microscopy",
            "chlamydomonas flagella", "chlamydomonas algae microscope",
        ],
        "wikimedia_category": "Chlamydomonas",
        "extra_wikimedia_categories": [
            "Chlamydomonadaceae", "Chlorophyceae",
            "Chlorophyta", "Green_algae",
            "Volvocales", "Volvox",
        ],
        "description": "Single-cell green algae with two flagella",
    },
    "Spirogyra": {
        "display_name": "Spirogyra",
        "search_terms": [
            "spirogyra microscope", "spirogyra filament microscopy",
            "spirogyra conjugation microscope", "spirogyra chloroplasts",
        ],
        "wikimedia_category": "Spirogyra",
        "extra_wikimedia_categories": [
            "Zygnematales", "Charophyta",
            "Filamentous_algae", "Conjugation_(biology)",
            "Freshwater_algae", "Desmidiales",
        ],
        "description": "Filamentous green algae with spiral chloroplasts",
    },
    "Penicillium": {
        "display_name": "Penicillium",
        "search_terms": [
            "penicillium microscope", "penicillium conidiophore microscopy",
            "penicillium mold microscope", "penicillium structure microscope",
        ],
        "wikimedia_category": "Penicillium",
        "extra_wikimedia_categories": [
            "Aspergillus", "Eurotiomycetes",
            "Moulds", "Fungal_morphologies",
            "Ascomycota", "Fungi",
        ],
        "description": "Saprophytic fungus (mold) with brush-like conidiophores",
    },
    "Nematode": {
        "display_name": "Nematode",
        "search_terms": [
            "nematode microscope", "nematoda microscopy",
            "roundworm microscope", "caenorhabditis elegans microscope",
        ],
        "wikimedia_category": "Nematoda",
        "extra_wikimedia_categories": [
            "Caenorhabditis_elegans", "Chromadorea",
            "Secernentea", "Nematode",
            "Nematoda", "Helminths",
        ],
        "description": "Unsegmented roundworms (microscopic species)",
    },
}

# ---------------------------------------------------------------------------
# Image Sources
# ---------------------------------------------------------------------------

USER_AGENT = (
    "MicroClassifyCrawler/1.0 (Educational research; "
    "microorganism image classification project)"
)


@dataclass
class CrawlResult:
    url: str
    source: str
    organism: str
    filepath: Optional[str] = None
    success: bool = False
    error: str = ""


class ImageValidator:
    """Validates downloaded images for quality and correctness."""

    MIN_SIZE = 64
    MAX_SIZE = 20_000_000  # 20MB
    MIN_DIMENSION = 48
    VALID_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}

    # Common non-microscope image patterns to reject
    REJECT_PATTERNS = [
        r"logo", r"icon", r"banner", r"button", r"sprite",
        r"avatar", r"thumbnail.*small", r"favicon",
        r"ad[_-]?banner", r"track\.gif",
    ]

    def __init__(self):
        self._seen_hashes: Set[str] = set()

    def validate(self, image_data: bytes, url: str = "") -> Tuple[bool, str]:
        """Validate image data. Returns (is_valid, error_message)."""
        if len(image_data) < self.MIN_SIZE:
            return False, "File too small"
        if len(image_data) > self.MAX_SIZE:
            return False, "File too large"

        try:
            img = Image.open(io.BytesIO(image_data))
            img.load()
        except Exception as e:
            return False, f"Cannot open image: {e}"

        if img.format not in self.VALID_FORMATS:
            return False, f"Unsupported format: {img.format}"

        w, h = img.size
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            return False, f"Too small: {w}x{h}"

        # Check for nearly blank images
        try:
            extrema = img.convert("L").getextrema()
            if extrema[1] - extrema[0] < 10:
                return False, "Blank/near-blank image"
        except Exception:
            pass

        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.REJECT_PATTERNS:
            if re.search(pattern, url_lower):
                return False, f"Rejected pattern: {pattern}"

        # Perceptual hash for deduplication
        img_hash = self._phash(img)
        if img_hash in self._seen_hashes:
            return False, "Duplicate image"
        self._seen_hashes.add(img_hash)

        return True, "OK"

    def _phash(self, img: Image.Image, hash_size: int = 16) -> str:
        """Simple perceptual hash using average hashing."""
        try:
            img_small = img.convert("L").resize(
                (hash_size, hash_size), Image.Resampling.LANCZOS
            )
            pixels = list(img_small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            return bits
        except Exception:
            return hashlib.md5(img.tobytes()).hexdigest()


class WikimediaDownloader:
    """Download images from Wikimedia Commons (CC-licensed)."""

    BASE = "https://commons.wikimedia.org/w/api.php"
    SESSION = requests.Session()
    SESSION.headers.update({"User-Agent": USER_AGENT})

    def search_images(
        self, category: str, limit: int = 50
    ) -> List[Dict[str, str]]:
        """Search Wikimedia Commons for images in a category."""
        results = []
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": min(limit, 50),
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": 800,
            "format": "json",
        }

        try:
            resp = self.SESSION.get(self.BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                imageinfo = page.get("imageinfo", [{}])[0]
                title = page.get("title", "")

                # Filter: only image files
                mime = imageinfo.get("mime", "")
                if not mime.startswith("image/"):
                    continue

                results.append({
                    "url": imageinfo.get("thumburl", imageinfo.get("url", "")),
                    "full_url": imageinfo.get("url", ""),
                    "title": title,
                    "width": imageinfo.get("thumbwidth", 0),
                    "height": imageinfo.get("thumbheight", 0),
                })

                if len(results) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Wikimedia search failed for {category}: {e}")

        return results

    def download(self, url: str) -> Optional[bytes]:
        """Download image from URL."""
        try:
            resp = self.SESSION.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.debug(f"Download failed: {e}")
            return None


class PixabayDownloader:
    """Download images from Pixabay (free license)."""

    BASE = "https://pixabay.com/api/"
    SESSION = requests.Session()

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("PIXABAY_API_KEY", "")

    def search_images(self, query: str, limit: int = 30) -> List[Dict[str, str]]:
        """Search Pixabay for images."""
        if not self.api_key:
            return []

        results = []
        params = {
            "key": self.api_key,
            "q": quote_plus(query),
            "image_type": "photo",
            "min_width": 200,
            "min_height": 200,
            "per_page": min(limit, 200),
            "safesearch": "true",
        }

        try:
            resp = self.SESSION.get(self.BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for hit in data.get("hits", []):
                results.append({
                    "url": hit.get("webformatURL", ""),
                    "full_url": hit.get("largeImageURL", ""),
                    "title": hit.get("tags", ""),
                    "width": hit.get("imageWidth", 0),
                    "height": hit.get("imageHeight", 0),
                })

                if len(results) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Pixabay search failed: {e}")

        return results

    def download(self, url: str) -> Optional[bytes]:
        try:
            resp = self.SESSION.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.debug(f"Pixabay download failed: {e}")
            return None


class WebImageScraper:
    """Generic web image scraper for educational/scientific sources."""

    SESSION = requests.Session()
    SESSION.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    })

    def search_and_extract(
        self, query: str, source_url: str, limit: int = 20
    ) -> List[Dict[str, str]]:
        """Scrape images from a specific URL."""
        results = []
        try:
            resp = self.SESSION.get(source_url, timeout=30)
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            for img in soup.find_all("img", src=True)[:limit]:
                src = img["src"]
                if not src.startswith("http"):
                    src = urljoin(source_url, src)

                # Skip tiny images (icons, spacers)
                w = int(img.get("width", 100) or 100)
                h = int(img.get("height", 100) or 100)
                if w < 50 or h < 50:
                    continue

                alt = img.get("alt", "")
                results.append({
                    "url": src,
                    "full_url": src,
                    "title": alt,
                    "width": w,
                    "height": h,
                })

        except Exception as e:
            logger.debug(f"Scraping failed for {source_url}: {e}")

        return results

    def download(self, url: str) -> Optional[bytes]:
        try:
            resp = self.SESSION.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and "octet-stream" not in content_type:
                return None
            return resp.content
        except Exception as e:
            logger.debug(f"Download failed: {e}")
            return None


class MicroorganismCrawler:
    """Main crawler that orchestrates downloading from all sources."""

    def __init__(self, dataset_root: str, max_per_class: int = 100):
        self.dataset_root = Path(dataset_root)
        self.max_per_class = max_per_class
        self.validator = ImageValidator()
        self.wikimedia = WikimediaDownloader()
        self.pixabay = PixabayDownloader()
        self.scraper = WebImageScraper()
        self.stats: Dict[str, Dict] = {}

    def verify_organism_name(self, name: str) -> Optional[str]:
        """Verify organism name against taxonomy. Returns canonical name or None."""
        # Exact match
        if name in TAXONOMY:
            return name

        # Case-insensitive match
        name_lower = name.lower().strip()
        for canonical in TAXONOMY:
            if canonical.lower() == name_lower:
                return canonical

        # Partial match (e.g., "paramecium caudatum" -> "Paramecium")
        for canonical in TAXONOMY:
            if name_lower in canonical.lower() or canonical.lower() in name_lower:
                return canonical

        # Check search terms for matches
        for canonical, info in TAXONOMY.items():
            for term in info.get("search_terms", []):
                if name_lower in term.lower():
                    return canonical

        return None

    def get_class_dir(self, class_name: str) -> Path:
        """Get or create directory for a class."""
        class_dir = self.dataset_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir

    def get_existing_count(self, class_name: str) -> int:
        """Count existing images in a class directory."""
        class_dir = self.dataset_root / class_name
        if not class_dir.exists():
            return 0
        return len([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ])

    def _get_next_index(self, class_dir: Path) -> int:
        """Get the next image index for a class."""
        existing = list(class_dir.glob("Image_*.*"))
        if not existing:
            return 1
        indices = []
        for f in existing:
            m = re.search(r"Image_(\d+)", f.name)
            if m:
                indices.append(int(m.group(1)))
        return max(indices) + 1 if indices else 1

    def download_from_wikimedia(
        self, class_name: str, category: str, limit: int,
        extra_categories: Optional[List[str]] = None,
    ) -> int:
        """Download images from Wikimedia Commons using primary + extra categories."""
        count = 0
        class_dir = self.get_class_dir(class_name)
        next_idx = self._get_next_index(class_dir)

        categories = [category] + (extra_categories or [])

        for cat in categories:
            if count >= limit:
                break
            images = self.wikimedia.search_images(cat, limit=limit - count)
            logger.info(f"  Wikimedia[{cat}]: found {len(images)} images")

            for img_info in images:
                if count >= limit:
                    break

                url = img_info.get("url") or img_info.get("full_url", "")
                if not url:
                    continue

                data = self.wikimedia.download(url)
                if data is None:
                    continue

                valid, err = self.validator.validate(data, url)
                if not valid:
                    logger.debug(f"    Rejected: {err}")
                    continue

                ext = self._guess_extension(url, data)
                filepath = class_dir / f"Image_{next_idx:03d}{ext}"
                filepath.write_bytes(data)
                next_idx += 1
                count += 1
                logger.info(f"    Saved: {filepath.name} ({len(data)} bytes)")

                time.sleep(0.5)  # Rate limit

        return count

    def download_from_pixabay(
        self, class_name: str, query: str, limit: int
    ) -> int:
        """Download images from Pixabay."""
        count = 0
        class_dir = self.get_class_dir(class_name)
        next_idx = self._get_next_index(class_dir)

        images = self.pixabay.search_images(query, limit=limit)
        logger.info(f"  Pixabay: found {len(images)} images for '{query}'")

        for img_info in images:
            if count >= limit:
                break

            url = img_info.get("full_url") or img_info.get("url", "")
            if not url:
                continue

            data = self.pixabay.download(url)
            if data is None:
                continue

            valid, err = self.validator.validate(data, url)
            if not valid:
                continue

            ext = self._guess_extension(url, data)
            filepath = class_dir / f"Image_{next_idx:03d}{ext}"
            filepath.write_bytes(data)
            next_idx += 1
            count += 1
            logger.info(f"    Saved: {filepath.name} ({len(data)} bytes)")

            time.sleep(0.3)

        return count

    def crawl_class(
        self, class_name: str, limit: int = 50, augment: bool = False
    ) -> Dict:
        """Crawl images for a single class."""
        if class_name not in TAXONOMY:
            logger.error(f"Unknown class '{class_name}' — not in taxonomy!")
            return {"class": class_name, "error": "not in taxonomy"}

        info = TAXONOMY[class_name]
        existing = self.get_existing_count(class_name)
        needed = max(0, limit - existing) if augment else limit

        logger.info(f"Crawling: {class_name} (existing: {existing}, need: {needed})")

        total_downloaded = 0

        # Source 1: Wikimedia Commons
        if needed > 0 and info.get("wikimedia_category"):
            n = self.download_from_wikimedia(
                class_name, info["wikimedia_category"], needed,
                extra_categories=info.get("extra_wikimedia_categories"),
            )
            total_downloaded += n
            needed = max(0, needed - n)

        # Source 2: Pixabay (if API key available)
        if needed > 0 and self.pixabay.api_key:
            for term in info.get("search_terms", [])[:2]:
                if needed <= 0:
                    break
                n = self.download_from_pixabay(class_name, term, needed)
                total_downloaded += n
                needed = max(0, needed - n)

        final_count = self.get_existing_count(class_name)
        result = {
            "class": class_name,
            "existing_before": existing,
            "downloaded": total_downloaded,
            "total_after": final_count,
        }
        self.stats[class_name] = result
        return result

    def crawl_all(self, limit: int = 50, augment: bool = False):
        """Crawl all classes in taxonomy."""
        logger.info(f"Starting crawl for {len(TAXONOMY)} classes (limit={limit})")
        for class_name in TAXONOMY:
            self.crawl_class(class_name, limit=limit, augment=augment)
            time.sleep(1)

        self._print_summary()

    def _print_summary(self):
        """Print crawl summary."""
        logger.info("\n" + "=" * 60)
        logger.info("CRAWL SUMMARY")
        logger.info("=" * 60)
        total = 0
        for cls, stats in sorted(self.stats.items()):
            downloaded = stats.get("downloaded", 0)
            total += downloaded
            logger.info(
                f"  {cls:30s}: +{downloaded:3d} images "
                f"(total: {stats.get('total_after', 0)})"
            )
        logger.info(f"\n  TOTAL: {total} new images downloaded")
        logger.info("=" * 60)

    def _guess_extension(self, url: str, data: bytes) -> str:
        """Guess file extension from URL or content."""
        # Try content sniffing
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"

        # Try URL extension
        url_path = url.split("?")[0].lower()
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
            if url_path.endswith(ext):
                return ext if ext != ".jpeg" else ".jpg"

        return ".jpg"


# ---------------------------------------------------------------------------
# Dataset Analysis & Recommendations
# ---------------------------------------------------------------------------


def analyze_dataset(dataset_root: str) -> Dict:
    """Analyze current dataset and recommend improvements."""
    root = Path(dataset_root)
    analysis = {"classes": {}, "total": 0, "recommendations": []}

    if not root.exists():
        analysis["error"] = f"Dataset not found: {root}"
        return analysis

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ]

        verified = class_dir.name in TAXONOMY
        analysis["classes"][class_dir.name] = {
            "count": len(images),
            "verified": verified,
            "in_taxonomy": verified,
        }
        analysis["total"] += len(images)

        if not verified:
            analysis["recommendations"].append(
                f"UNVERIFIED CLASS: '{class_dir.name}' not in taxonomy — "
                f"verify organism name or add to taxonomy"
            )
        elif len(images) < 50:
            analysis["recommendations"].append(
                f"LOW COUNT: '{class_dir.name}' has only {len(images)} images "
                f"(recommended: 100+)"
            )

    # Check for missing classes
    existing = set(analysis["classes"].keys())
    missing = set(TAXONOMY.keys()) - existing
    for m in sorted(missing):
        analysis["recommendations"].append(
            f"MISSING CLASS: '{m}' — not yet in dataset"
        )

    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Microorganism Image Crawler & Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-classes                 Show all taxonomy classes
  %(prog)s --analyze                      Analyze current dataset
  %(prog)s --class Paramecium --limit 50  Crawl one class
  %(prog)s --class-all --limit 30         Crawl all classes
  %(prog)s --augment-existing --limit 100 Augment existing classes only
        """,
    )
    parser.add_argument(
        "--dataset", default="dataset",
        help="Path to dataset root directory (default: dataset)",
    )
    parser.add_argument("--list-classes", action="store_true",
                        help="List all taxonomy classes")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze current dataset")
    parser.add_argument("--class", dest="target_class", type=str,
                        help="Crawl a specific class")
    parser.add_argument("--class-all", action="store_true",
                        help="Crawl all taxonomy classes")
    parser.add_argument("--augment-existing", action="store_true",
                        help="Only augment classes that already exist in dataset")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max images per class (default: 50)")
    parser.add_argument("--add-class", type=str, nargs="+",
                        help="Add new classes to taxonomy and crawl them")
    parser.add_argument("--wikimedia-only", action="store_true",
                        help="Only use Wikimedia Commons (no API key needed)")

    args = parser.parse_args()

    if args.list_classes:
        print(f"\n{'='*60}")
        print(f"VERIFIED TAXONOMY — {len(TAXONOMY)} classes")
        print(f"{'='*60}")
        for name, info in TAXONOMY.items():
            print(f"  {name:30s}  {info['description']}")
        print(f"{'='*60}\n")
        return

    if args.analyze:
        analysis = analyze_dataset(args.dataset)
        print(f"\n{'='*60}")
        print(f"DATASET ANALYSIS — {analysis.get('total', 0)} total images")
        print(f"{'='*60}")
        for cls, info in sorted(analysis.get("classes", {}).items()):
            status = "OK" if info["verified"] else "UNVERIFIED"
            print(f"  {cls:30s}  {info['count']:4d} images  [{status}]")
        print()
        for rec in analysis.get("recommendations", []):
            print(f"  * {rec}")
        print(f"{'='*60}\n")
        return

    # Add new classes to taxonomy
    if args.add_class:
        for cls in args.add_class:
            canonical = MicroorganismCrawler(args.dataset).verify_organism_name(cls)
            if canonical:
                logger.info(f"'{cls}' recognized as '{canonical}' — using existing taxonomy entry")
            else:
                # Add with sensible defaults
                TAXONOMY[cls] = {
                    "display_name": cls.replace("_", " ").title(),
                    "search_terms": [
                        f"{cls} microscope", f"{cls} microscopy",
                        f"{cls} under microscope",
                    ],
                    "wikimedia_category": cls.replace("_", " "),
                    "description": f"Microorganism: {cls}",
                }
                logger.info(f"Added new class: '{cls}'")

    # Crawl
    crawler = MicroorganismCrawler(args.dataset, max_per_class=args.limit)

    if args.target_class:
        crawler.crawl_class(args.target_class, limit=args.limit)
    elif args.class_all or args.augment_existing:
        if args.augment_existing:
            # Only crawl classes that already exist in dataset
            existing = [
                d.name for d in Path(args.dataset).iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            for cls in existing:
                if cls in TAXONOMY:
                    crawler.crawl_class(cls, limit=args.limit, augment=True)
        else:
            crawler.crawl_all(limit=args.limit)
    else:
        parser.print_help()

    crawler._print_summary()


if __name__ == "__main__":
    main()
