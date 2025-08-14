#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import io
import os
import re
import sys
import time
import zipfile
import uuid
import json
import threading
import traceback
from datetime import datetime, timezone
from threading import Lock
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Protocol, Sequence, Callable, Dict, Any
from urllib.parse import urlparse, urljoin, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_file, render_template, g
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

# -----------------------------
# Config / App
# -----------------------------

load_dotenv()
API_PASSWORD = os.getenv("API_PASSWORD")
API_URL = os.getenv("API_URL") or "https://translate.aaroncollins.info/translate-image"
JOBS_DIR = Path(os.getenv("JOB_DIR", "./jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

if not API_PASSWORD:
    print("ERROR: API_PASSWORD is not set (env or .env).", file=sys.stderr)

app = Flask(__name__)

# -----------------------------
# Locks per job (status.json safety)
# -----------------------------

STATUS_LOCKS: dict[str, Lock] = {}

def _get_lock(job_id: str) -> Lock:
    if job_id not in STATUS_LOCKS:
        STATUS_LOCKS[job_id] = Lock()
    return STATUS_LOCKS[job_id]

# -----------------------------
# Utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(msg: str) -> None:
    print(msg, flush=True)

def sanitize_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def per_url_dir(base_dir: Path, url: str) -> Path:
    host = urlparse(url).netloc.replace(":", "_").lower()
    path_parts = [p for p in urlparse(url).path.split("/") if p]
    id_part = None
    if "mangadex.org" in host and len(path_parts) >= 5 and path_parts[3] == "chapter":
        id_part = path_parts[4]
    elif path_parts:
        id_part = path_parts[-1]
    safe = sanitize_id(id_part or str(int(time.time())))
    out = base_dir / f"{host}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def read_urls_from_text(s: str) -> List[str]:
    return [line.strip() for line in s.splitlines() if line.strip()]

_index_re = re.compile(r"(\d{3,})")

def extract_index_from_name(name: str) -> int:
    m = _index_re.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    digits = re.findall(r"\d+", name)
    if digits:
        try:
            return int(digits[-1])
        except Exception:
            pass
    return 10**9

def stitch_vertically(pil_images: List[Image.Image]) -> Image.Image:
    widths = [im.width for im in pil_images]
    heights = [im.height for im in pil_images]
    max_w = max(widths) if widths else 1
    total_h = sum(heights) if heights else 1
    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in pil_images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        canvas.paste(im, (0, y))
        y += im.height
    return canvas

# -----------------------------
# Status & Error Recording
# -----------------------------

def _safe_json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        # Corrupted status file: keep a backup and start fresh
        backup = path.with_suffix(".corrupt.json")
        try:
            backup.write_text(path.read_text())
        except Exception:
            pass
        return {}

def update_status(job_id, **kwargs):
    status_file = JOBS_DIR / job_id / "status.json"
    lock = _get_lock(job_id)
    with lock:
        status = _safe_json_read(status_file)

        for key, value in kwargs.items():
            if key == "jobs" and isinstance(value, dict):
                status.setdefault("jobs", {})
                # deep-merge per-URL dicts
                for u, meta in value.items():
                    if isinstance(meta, dict) and isinstance(status["jobs"].get(u), dict):
                        status["jobs"][u].update(meta)
                    else:
                        status["jobs"][u] = meta
            elif key == "errors" and isinstance(value, list):
                # append error records
                status.setdefault("errors", [])
                status["errors"].extend(value)
            elif isinstance(value, dict) and isinstance(status.get(key), dict):
                status[key].update(value)
            else:
                status[key] = value

        status_file.write_text(json.dumps(status, indent=2))

def record_error(
    job_id: Optional[str],
    *,
    scope: str,           # e.g. "api", URL string, "worker:<url>", "zip", etc.
    code: str,            # machine-readable code
    message: str,         # human message
    where: Optional[str] = None,   # function or component
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None
) -> None:
    err: Dict[str, Any] = {
        "ts": now_iso(),
        "scope": scope,
        "code": code,
        "message": message,
    }
    if where:
        err["where"] = where
    if request_id:
        err["request_id"] = request_id
    if extra:
        err["extra"] = extra
    if exc:
        err["exception_type"] = exc.__class__.__name__
        err["traceback"] = "".join(traceback.format_exception(exc)).strip()

    if job_id:
        update_status(job_id, errors=[err])
    else:
        # no job id: write to a global file
        global_err = JOBS_DIR / "global_errors.jsonl"
        with open(global_err, "a", encoding="utf-8") as f:
            f.write(json.dumps(err) + "\n")

# -----------------------------
# Strategy interfaces (protocol)
# -----------------------------

class ScraperStrategy(Protocol):
    def supports(self, url: str) -> bool: ...
    def download_pages(
        self,
        url: str,
        out_dir: Path,
        session: requests.Session,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ) -> List[Path]: ...

# -----------------------------
# Generic scraper framework
# -----------------------------

@dataclass(frozen=True)
class PageItem:
    """
    A single downloadable page for a chapter. `index` is 1-based and used to order pages.
    """
    url: str
    index: int

class GenericScraper(ScraperStrategy):
    """
    Base class that implements the "download pages" pipeline.
    Subclasses implement:
      - `supports(url) -> bool` (usually by domain match)
      - `get_pages(url, session) -> Sequence[PageItem]`
      - Optional: override `file_prefix(url) -> str` to control filenames.
    """
    name: str = "page"

    def supports(self, url: str) -> bool:
        raise NotImplementedError

    def get_pages(self, url: str, session: requests.Session) -> Sequence[PageItem]:
        raise NotImplementedError

    def file_prefix(self, url: str) -> str:
        """Filename prefix (default: self.name). Override to include IDs (e.g. chapter id)."""
        return self.name

    def download_pages(
        self,
        url: str,
        out_dir: Path,
        session: requests.Session,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ) -> List[Path]:
        pages = list(self.get_pages(url, session))
        prefix = self.file_prefix(url)
        paths: List[Path] = []
        for item in pages:
            fname = f"{prefix}_{int(item.index):03d}.png"
            out_path = out_dir / fname
            try:
                r = session.get(item.url, timeout=60)
                r.raise_for_status()
                out_path.write_bytes(r.content)
                paths.append(out_path)
            except Exception as e:
                if on_error:
                    try:
                        on_error(item.url, e)
                    except Exception:
                        pass
                log(f"[{self.__class__.__name__}] ERROR downloading {item.url}: {e}")
        return paths

# -----------------------------
# Scraper registry (plug-and-play)
# -----------------------------

_SCRAPER_REGISTRY: List[GenericScraper] = []

def register_scraper(scraper_cls: type[GenericScraper]) -> type[GenericScraper]:
    _SCRAPER_REGISTRY.append(scraper_cls())  # instantiate once (stateless)
    return scraper_cls

# -----------------------------
# Site scrapers (converted to GenericScraper)
# -----------------------------

@register_scraper
class RawKumaScraper(GenericScraper):
    name = "rawkuma"

    def supports(self, url: str) -> bool:
        return "rawkuma" in urlparse(url).netloc.lower()

    def get_pages(self, url: str, session: requests.Session) -> Sequence[PageItem]:
        log(f"[RawKuma] GET {url}")
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        reader = soup.find("div", id="readerarea")
        if not reader:
            return []

        # Prefer explicit data-index; fallback to visible <img> nodes
        imgs = reader.find_all("img", attrs={"data-index": True})
        if not imgs:
            imgs = [
                img for img in reader.find_all("img")
                if (img.get("data-src") or img.get("src")) and "placeholder" not in (img.get("src") or "")
            ]

        items: List[PageItem] = []
        for idx, tag in enumerate(imgs):
            src = tag.get("data-src") or tag.get("src")
            if not src:
                continue
            page_num = int(tag.get("data-index") or idx) + 1
            img_url = urljoin(url, src)
            items.append(PageItem(url=img_url, index=page_num))

        items.sort(key=lambda p: p.index)
        return items

@register_scraper
class MangaDexScraper(GenericScraper):
    name = "chapter"

    def supports(self, url: str) -> bool:
        return "mangadex.org" in urlparse(url).netloc.lower()

    def _extract_chapter_id(self, url: str) -> Optional[str]:
        parts = [p for p in url.strip().split("/") if p]
        try:
            idx = parts.index("chapter")
            return parts[idx + 1]
        except Exception:
            try:
                return parts[4]
            except Exception:
                return None

    def file_prefix(self, url: str) -> str:
        # Include chapter id to preserve your original filename style
        ch_id = self._extract_chapter_id(url) or "chapter"
        return f"chapter_{ch_id}"

    def get_pages(self, url: str, session: requests.Session) -> Sequence[PageItem]:
        chapter_id = self._extract_chapter_id(url)
        if not chapter_id:
            log(f"[MangaDex] ERROR parsing chapter id: {url}")
            return []

        api_url = f"https://api.mangadex.org/at-home/server/{chapter_id}?forcePort443=false"
        r = session.get(api_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("result") != "ok":
            return []

        base_url = data["baseUrl"]
        ch = data["chapter"]
        hash_val = ch["hash"]
        pages = ch["data"]

        items: List[PageItem] = []
        for i, filename in enumerate(pages, start=1):
            img_url = f"{base_url}/data/{hash_val}/{filename}"
            items.append(PageItem(url=img_url, index=i))
        return items

@register_scraper
class UsagiOneScraper(GenericScraper):
    """
    Scraper for web.usagi.one, e.g.:
    https://web.usagi.one/31921/vol3/14.1

    Rules:
      - Find #fotocontext
      - For each div.manga-img-placeholder, take img.manga-img
      - src may be in src / data-src / data-lazy-src / srcset
    """
    name = "usagi"

    def supports(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return "usagi.one" in host

    def _get_img_src(self, tag) -> Optional[str]:
        for key in ("data-src", "data-lazy-src", "src"):
            val = tag.get(key)
            if val:
                return val.strip()
        srcset = tag.get("srcset")
        if srcset:
            first = srcset.split(",")[0].strip().split(" ")[0]
            if first:
                return first
        return None

    def get_pages(self, url: str, session: requests.Session) -> Sequence[PageItem]:
        log(f"[UsagiOne] GET {url}")
        r = session.get(url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        root = soup.find(id="fotocontext")
        if not root:
            log("[UsagiOne] WARN: #fotocontext not found")
            return []

        containers = root.select("div.manga-img-placeholder")
        if not containers:
            # Fallback: any img.manga-img under fotocontext
            containers = root.select("img.manga-img")

        items: List[PageItem] = []
        idx = 1
        for node in containers:
            img = node if getattr(node, "name", None) == "img" else node.find("img", class_="manga-img")
            if not img:
                continue
            src = self._get_img_src(img)
            if not src:
                continue
            img_url = urljoin(url, src)
            items.append(PageItem(url=img_url, index=idx))
            idx += 1
        return items

    # Add Referer during file GETs (common requirement for readers)
    def download_pages(
        self,
        url: str,
        out_dir: Path,
        session: requests.Session,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ) -> List[Path]:
        old_ref = session.headers.get("Referer")
        session.headers["Referer"] = url
        try:
            return super().download_pages(url, out_dir, session, on_error=on_error)
        finally:
            if old_ref is None:
                session.headers.pop("Referer", None)
            else:
                session.headers["Referer"] = old_ref

# -----------------------------
# Translator API
# -----------------------------

@dataclass
class TranslatorAPI:
    api_url: str
    api_key: str
    target_lang: str = "en"
    timeout: int = 60
    max_retries: int = 3
    backoff_base: float = 1.5

    def translate_image(self, image_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Returns (translated_png_bytes, error_message). On success, error_message is None.
        """
        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                files = {"file": ("page.png", image_bytes, "image/png")}
                data = {"target": self.target_lang}
                headers = {"X-API-KEY": self.api_key}
                resp = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                b64_png = body.get("translated_image", "")
                if "," in b64_png:
                    b64_png = b64_png.split(",", 1)[1]
                if not b64_png:
                    last_err = "empty_translated_image"
                    time.sleep(self.backoff_base ** attempt)
                    continue
                return base64.b64decode(b64_png), None
            except Exception as e:
                # capture HTTP details if available
                status = getattr(e, "response", None).status_code if hasattr(e, "response") and e.response else None
                last_err = f"{e.__class__.__name__}: {str(e)}"
                if status:
                    last_err += f" (status={status})"
                time.sleep(self.backoff_base ** attempt)
        return None, last_err

# -----------------------------
# Job Runner with Status Tracking
# -----------------------------

@dataclass
class JobConfig:
    out_base: Path
    pages_workers: int
    rotate_90: bool = False
    target_lang: str = "en"
    timeout: int = 60
    retries: int = 3

def process_url(
    url,
    scrapers,
    translator: TranslatorAPI,
    cfg,
    session,
    job_id=None
):
    out_dir = per_url_dir(cfg.out_base, url)
    # record folder (relative) early for downloads
    if job_id:
        folder_rel = str(out_dir.relative_to(cfg.out_base))
        update_status(job_id, jobs={url: {"folder": folder_rel}})

    strategy = next((s for s in scrapers if s.supports(url)), None)
    if not strategy:
        if job_id:
            record_error(job_id, scope=url, code="unsupported_host", message="No scraper supports this URL", where="process_url")
            update_status(job_id, jobs={url: {"status": "failed", "error": "unsupported_host"}})
        return

    try:
        # Download pages (with per-image error callback)
        def dl_err(img_url: str, exc: Exception):
            record_error(job_id, scope=url, code="download_failed",
                         message=f"Failed to download page image",
                         where="GenericScraper.download_pages",
                         extra={"img_url": img_url},
                         exc=exc)

        originals = strategy.download_pages(url, out_dir, session, on_error=dl_err)
        total_pages = len(originals)

        # Initialize per-URL tracking
        if job_id:
            update_status(job_id, jobs={url: {"progress": 0, "total": total_pages, "status": "running"}})

        # No pages -> treat as failed (retryable)
        if total_pages == 0:
            if job_id:
                status_file = JOBS_DIR / job_id / "status.json"
                lock = _get_lock(job_id)
                with lock:
                    status = _safe_json_read(status_file)
                    status.setdefault("jobs", {}).setdefault(url, {})
                    status["jobs"][url].update({"status": "failed", "error": "no_pages"})
                    status_file.write_text(json.dumps(status, indent=2))
                record_error(job_id, scope=url, code="no_pages", message="Scraper returned 0 pages", where="process_url")
            return

        def work(page_path: Path):
            try:
                index = extract_index_from_name(page_path.name)
                img_bytes = page_path.read_bytes()
                translated_bytes, xerr = translator.translate_image(img_bytes)

                # If translation failed, fall back to original but record error
                used_bytes = translated_bytes or img_bytes
                if xerr:
                    record_error(job_id, scope=f"worker:{url}", code="translate_failed",
                                 message=xerr, where="TranslatorAPI.translate_image",
                                 extra={"page_index": index, "file": page_path.name})

                # Load and post-process
                im = Image.open(BytesIO(used_bytes))
                im.load()  # ensure the image is fully loaded before closing BytesIO
                if cfg.rotate_90:
                    im = im.transpose(Image.ROTATE_90)
                # Save translated (or original) page
                im.convert("RGB").save(out_dir / f"translated_{index:03d}.png", "PNG")
                return index, im
            except Exception as e:
                record_error(job_id, scope=f"worker:{url}", code="page_processing_error",
                             message="Failed to process page image", where="work",
                             extra={"file": page_path.name}, exc=e)
                raise

        completed_pages = 0
        images_indexed: List[Tuple[int, Image.Image]] = []
        errors_during_pages = False

        with ThreadPoolExecutor(max_workers=cfg.pages_workers) as ex:
            futures = [ex.submit(work, p) for p in originals]
            for fut in as_completed(futures):
                try:
                    idx, im = fut.result()
                    images_indexed.append((idx, im))
                    completed_pages += 1
                    if job_id:
                        status_file = JOBS_DIR / job_id / "status.json"
                        lock = _get_lock(job_id)
                        with lock:
                            status = _safe_json_read(status_file)
                            status.setdefault("jobs", {}).setdefault(url, {})
                            status["jobs"][url]["progress"] = completed_pages
                            status_file.write_text(json.dumps(status, indent=2))
                except Exception:
                    errors_during_pages = True
                    # progress already updated for other futures; continue

        if not images_indexed:
            # If every page failed, mark failed
            if job_id:
                status_file = JOBS_DIR / job_id / "status.json"
                lock = _get_lock(job_id)
                with lock:
                    status = _safe_json_read(status_file)
                    status.setdefault("jobs", {}).setdefault(url, {})
                    status["jobs"][url].update({"status": "failed", "error": "all_pages_failed"})
                    status_file.write_text(json.dumps(status, indent=2))
            record_error(job_id, scope=url, code="all_pages_failed",
                         message="All page workers failed", where="process_url")
            return

        # Save stitched image
        try:
            images_indexed.sort(key=lambda t: t[0])
            stitched = stitch_vertically([im for _, im in images_indexed])
            stitched.save(out_dir / f"translated_final.png", "PNG")
        except Exception as e:
            record_error(job_id, scope=url, code="stitch_failed",
                         message="Failed to stitch final image", where="stitch_vertically", exc=e)
            # Even if stitching fails, continue to mark finished if at least one page saved

        # Mark URL job finished (SUCCESS) and increment overall URL count
        if job_id:
            status_file = JOBS_DIR / job_id / "status.json"
            lock = _get_lock(job_id)
            with lock:
                status = _safe_json_read(status_file)
                status.setdefault("jobs", {}).setdefault(url, {})
                status["jobs"][url]["status"] = "finished" if not errors_during_pages else "finished_with_warnings"
                if "overall" in status and "progress" in status["overall"]:
                    status["overall"]["progress"] += 1
                status_file.write_text(json.dumps(status, indent=2))

    except Exception as e:
        # Mark URL failed (so it’s retryable)
        if job_id:
            status_file = JOBS_DIR / job_id / "status.json"
            lock = _get_lock(job_id)
            with lock:
                status = _safe_json_read(status_file)
                status.setdefault("jobs", {}).setdefault(url, {"progress": 0, "total": 0})
                status["jobs"][url].update({"status": "failed", "error": str(e)})
                status_file.write_text(json.dumps(status, indent=2))
        record_error(job_id, scope=url, code="unhandled_exception",
                     message="Unhandled exception while processing URL", where="process_url", exc=e)
        return

def run_translation_job(job_id, urls, batch_workers, page_workers, target_lang, rotate_90, timeout, retries):
    job_root = JOBS_DIR / job_id
    jobs_info = {u: {"progress": 0, "total": 0, "status": "queued"} for u in urls}
    # overall = URL-based
    update_status(job_id, status="running", overall={"progress": 0, "total": len(urls)}, jobs=jobs_info)
    translator = TranslatorAPI(api_url=API_URL, api_key=API_PASSWORD, target_lang=target_lang, timeout=timeout, max_retries=retries)
    # Use registry (plug-and-play)
    strategies: List[ScraperStrategy] = list(_SCRAPER_REGISTRY)
    cfg = JobConfig(out_base=job_root, pages_workers=page_workers, rotate_90=rotate_90, target_lang=target_lang, timeout=timeout, retries=retries)

    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "ScreenTranslatorAPI/1.0"})
            with ThreadPoolExecutor(max_workers=batch_workers) as ex:
                for _ in as_completed([ex.submit(process_url, u, strategies, translator, cfg, session, job_id) for u in urls]):
                    pass

        # Build final zip when everything is finished
        try:
            (job_root / "result.zip").write_bytes(zip_directory_to_bytes(job_root, exclude={"result.zip"}))
        except Exception as e:
            record_error(job_id, scope="zip", code="zip_build_failed",
                         message="Failed to build final result.zip", where="run_translation_job", exc=e)
        update_status(job_id, status="finished")
    except Exception as e:
        record_error(job_id, scope="job", code="run_failed",
                     message="Unhandled exception in run_translation_job", where="run_translation_job", exc=e)
        update_status(job_id, status="failed")

# -----------------------------
# Flask request context helpers
# -----------------------------

@app.before_request
def assign_request_id():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

@app.after_request
def attach_request_id(resp):
    # Always echo a request id for correlation
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return resp

def _json_error(code: int, message: str, *, job_id: Optional[str] = None, err_code: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
    payload = {
        "error": message,
        "code": err_code or "error",
        "request_id": getattr(g, "request_id", None),
        "ts": now_iso(),
    }
    if extra:
        payload["extra"] = extra
    if job_id:
        record_error(job_id, scope="api", code=payload["code"], message=message, where=request.endpoint or "api", request_id=payload["request_id"], extra=extra)
    return jsonify(payload), code

# -----------------------------
# API Endpoints
# -----------------------------

@app.route("/api/start-translation", methods=["POST"])
def start_translation():
    data = request.get_json(force=True, silent=True) or {}
    urls = [u.strip() for u in data.get("urls", []) if isinstance(u, str) and u.strip()]
    if not urls:
        return _json_error(400, "No URLs provided", err_code="missing_urls")

    job_id = str(uuid.uuid4())
    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    jobs_info = {u: {"progress": 0, "total": 0, "status": "queued"} for u in urls}

    # set overall total up-front to avoid 0/0 flicker
    update_status(job_id, status="queued", overall={"progress": 0, "total": len(urls)}, jobs=jobs_info)

    try:
        threading.Thread(
            target=run_translation_job,
            args=(job_id, urls, int(data.get("batch_workers", 10)), int(data.get("page_workers", 32)),
                  str(data.get("target_lang", "en")), bool(data.get("rotate_90", False)),
                  int(data.get("timeout", 60)), int(data.get("retries", 3))),
            daemon=True
        ).start()
    except Exception as e:
        record_error(job_id, scope="api", code="thread_start_failed",
                     message="Failed to start job thread", where="start_translation", exc=e)
        return _json_error(500, "Failed to start job", job_id=job_id, err_code="start_failed")

    return jsonify({"job_id": job_id, "request_id": g.request_id})

@app.route("/api/job-status/<job_id>")
def job_status(job_id):
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")
    lock = _get_lock(job_id)
    with lock:
        return jsonify(_safe_json_read(status_file))

@app.route("/api/download/<job_id>")
def download_result(job_id):
    zip_path = JOBS_DIR / job_id / "result.zip"
    if not zip_path.exists():
        return _json_error(404, "Result not ready", job_id=job_id, err_code="result_not_ready")
    return send_file(zip_path, as_attachment=True, download_name=f"{job_id}.zip")

@app.route("/api/download-finished/<job_id>")
def download_finished(job_id):
    """Zip only finished URL folders so far."""
    status_file = JOBS_DIR / job_id / "status.json"
    job_root = JOBS_DIR / job_id
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")

    lock = _get_lock(job_id)
    with lock:
        status = _safe_json_read(status_file)
        jobs = status.get("jobs", {})

    folders = []
    for url, meta in jobs.items():
        if meta.get("status") in ("finished", "finished_with_warnings") and meta.get("folder"):
            folder = job_root / meta["folder"]
            if folder.exists():
                folders.append(folder)

    if not folders:
        return _json_error(400, "No finished URLs yet", job_id=job_id, err_code="none_finished")

    try:
        buf = zip_selected_paths(job_root, folders, exclude={"result.zip", "status.json"})
        filename = f"{job_id}_finished_{int(time.time())}.zip"
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=filename)
    except Exception as e:
        record_error(job_id, scope="api", code="zip_finished_failed",
                     message="Failed to build finished-only zip", where="download_finished", exc=e)
        return _json_error(500, "Failed to build zip", job_id=job_id, err_code="zip_failed")

@app.route("/api/download-one/<job_id>")
def download_one(job_id):
    """Zip just one URL folder:
       Preferred:  /api/download-one/<job_id>?folder=<relative-folder>
       Fallback:   /api/download-one/<job_id>?u=<url-encoded-url>
    """
    job_root = JOBS_DIR / job_id
    status_file = job_root / "status.json"
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")

    folder_param = request.args.get("folder", "").strip()
    url_param = request.args.get("u", "").strip()

    # Load status once under lock
    lock = _get_lock(job_id)
    with lock:
        status = _safe_json_read(status_file)
        jobs = status.get("jobs", {})

    # Preferred: folder param
    if folder_param:
        target = (job_root / folder_param)
        if target.exists() and target.is_dir():
            try:
                buf = zip_selected_paths(job_root, [target], exclude={"result.zip", "status.json"})
                filename = f"{job_id}_one_{sanitize_id(folder_param)}.zip"
                return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=filename)
            except Exception as e:
                record_error(job_id, scope="api", code="zip_single_failed",
                             message="Failed to build single-folder zip", where="download_one",
                             extra={"folder": folder_param}, exc=e)
                return _json_error(500, "Failed to build zip", job_id=job_id, err_code="zip_failed")
        return _json_error(404, "Folder not found", err_code="folder_not_found")

    # Fallback: try url param with a couple of normalizations
    if url_param:
        # exact
        meta = jobs.get(url_param)
        # try strip trailing slash
        if not meta and url_param.endswith("/"):
            meta = jobs.get(url_param[:-1])
        # try add trailing slash
        if not meta and not url_param.endswith("/"):
            meta = jobs.get(url_param + "/")

        if meta and meta.get("folder"):
            target = job_root / meta["folder"]
            if target.exists():
                try:
                    buf = zip_selected_paths(job_root, [target], exclude={"result.zip", "status.json"})
                    filename = f"{job_id}_one_{sanitize_id(meta['folder'])}.zip"
                    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=filename)
                except Exception as e:
                    record_error(job_id, scope="api", code="zip_single_failed",
                                 message="Failed to build single-folder zip", where="download_one",
                                 extra={"url": url_param}, exc=e)
                    return _json_error(500, "Failed to build zip", job_id=job_id, err_code="zip_failed")
        return _json_error(404, "URL not found in job", err_code="url_not_found")

    return _json_error(400, "Missing ?folder=<id> or ?u=<url>", err_code="missing_params")

@app.route("/api/retry-url/<job_id>", methods=["POST"])
def retry_failed_urls(job_id):
    """Retry failed URLs for a job. Optional JSON body: {"urls": ["...","..."]} to retry specific URLs."""
    status_file = JOBS_DIR / job_id / "status.json"
    job_root = JOBS_DIR / job_id
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")

    # read status once
    lock = _get_lock(job_id)
    with lock:
        status = _safe_json_read(status_file)
        jobs = (status.get("jobs", {}) or {}).copy()  # shallow copy ok

    body = request.get_json(silent=True) or {}
    requested = body.get("urls")

    # allow retry of specified URLs if they exist & not currently running
    def eligible(u, meta):
        st = (meta or {}).get("status")
        return st in ("failed", "skipped", "finished_with_warnings")  # extend if you want

    if requested:
        retry_urls = [u for u in requested if u in jobs and eligible(u, jobs[u])]
    else:
        retry_urls = [u for u, meta in jobs.items() if eligible(u, meta)]

    if not retry_urls:
        return jsonify({"message": "No failed URLs to retry", "request_id": g.request_id}), 200

    # Prep runner bits
    translator = TranslatorAPI(api_url=API_URL, api_key=API_PASSWORD)
    strategies: List[ScraperStrategy] = list(_SCRAPER_REGISTRY)
    cfg = JobConfig(out_base=job_root, pages_workers=int(os.getenv("PAGE_WORKERS", "32")))

    # Mark retries as running (reset counters, KEEP folder)
    for u in retry_urls:
        prev = jobs.get(u, {}) or {}
        update_status(
            job_id,
            jobs={
                u: {
                    "folder": prev.get("folder"),          # <- preserve
                    "progress": 0,
                    "total": prev.get("total", 0),
                    "status": "running",
                    "error": None                           # clear last error
                }
            }
        )

    # Run retries
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "ScreenTranslatorAPI/1.0"})
            with ThreadPoolExecutor(max_workers=int(os.getenv("BATCH_WORKERS", "10"))) as ex:
                futs = [ex.submit(process_url, u, strategies, translator, cfg, session, job_id) for u in retry_urls]
                for _ in as_completed(futs):
                    pass
    except Exception as e:
        record_error(job_id, scope="api", code="retry_failed",
                     message="Failed while retrying URLs", where="retry_failed_urls", exc=e)
        return _json_error(500, "Retry failed", job_id=job_id, err_code="retry_failed")

    # Rebuild the package after retry (exclude result.zip to avoid nesting)
    try:
        (job_root / "result.zip").write_bytes(zip_directory_to_bytes(job_root, exclude={"result.zip"}))
    except Exception as e:
        record_error(job_id, scope="zip", code="zip_build_failed",
                     message="Failed to rebuild result.zip after retry", where="retry_failed_urls", exc=e)

    # Return updated status
    with lock:
        return jsonify(_safe_json_read(status_file))

@app.route("/api/preview/<job_id>")
def preview_image(job_id):
    """Serve the stitched PNG for a single URL folder."""
    job_root = JOBS_DIR / job_id
    status_file = job_root / "status.json"
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")

    folder = (request.args.get("folder") or "").strip()
    if not folder:
        return _json_error(400, "Missing folder", err_code="missing_folder")

    target = (job_root / folder / "translated_final.png")
    if not target.exists():
        return _json_error(404, "Preview not available", job_id=job_id, err_code="preview_missing")

    # Stream the PNG so the browser can display it
    return send_file(target, mimetype="image/png")

@app.route("/api/errors/<job_id>")
def job_errors(job_id):
    """
    Return the error log for a specific job.
    Response Example:
    {
      "job_id": "...",
      "errors": [ {ts, scope, code, message, ...}, ... ],
      "request_id": "..."
    }
    """
    status_file = JOBS_DIR / job_id / "status.json"
    if not status_file.exists():
        return _json_error(404, "Job not found", err_code="job_not_found")
    lock = _get_lock(job_id)
    with lock:
        status = _safe_json_read(status_file)
    return jsonify({
        "job_id": job_id,
        "errors": status.get("errors", []),
        "request_id": g.request_id
    })

@app.errorhandler(404)
def not_found(e):
    return _json_error(404, "Not found", err_code="not_found")

@app.errorhandler(405)
def method_not_allowed(e):
    return _json_error(405, "Method not allowed", err_code="method_not_allowed")

@app.errorhandler(Exception)
def unhandled(e):
    # Try to attach job_id if provided as param/path in common endpoints
    job_id = request.view_args.get("job_id") if request.view_args else None
    record_error(job_id, scope="api", code="unhandled_exception", message=str(e), where="errorhandler", exc=e, request_id=g.request_id)
    return _json_error(500, "Internal server error", job_id=job_id, err_code="internal_error")

@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------
# ZIP helpers
# -----------------------------

def zip_directory_to_bytes(base_dir: Path, exclude: set[str] | None = None) -> bytes:
    exclude = exclude or set()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f in exclude:
                    continue
                full = Path(root) / f
                arc = full.relative_to(base_dir)
                try:
                    zf.write(full, arcname=str(arc))
                except Exception as e:
                    # best-effort: log and continue
                    record_error(None, scope="zip", code="file_zip_failed",
                                 message=f"Failed to add file to zip: {str(arc)}",
                                 where="zip_directory_to_bytes", exc=e)
    buf.seek(0)
    return buf.getvalue()

def zip_selected_paths(base_dir: Path, folders: List[Path], exclude: set[str] | None = None) -> io.BytesIO:
    exclude = exclude or set()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f in exclude:
                        continue
                    full = Path(root) / f
                    arc = full.relative_to(base_dir)
                    try:
                        zf.write(full, arcname=str(arc))
                    except Exception as e:
                        record_error(None, scope="zip", code="file_zip_failed",
                                     message=f"Failed to add file to zip: {str(arc)}",
                                     where="zip_selected_paths", exc=e)
    buf.seek(0)
    return buf

# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
