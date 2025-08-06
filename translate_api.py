"""
Cloud Vision + Translation micro‑service (Compose‑only build, client‑supplied API‑key)
===================================================================================

**Change:** The client must now include `X-API-KEY: <password>` (or `?key=`) with every request—Caddy no longer injects it automatically. The 401 guard in `translate_app.py` remains unchanged.

Repo tree (unchanged):
```
translator/
├── translate_app.py      # Flask app with Vision → Translation chain (+401 auth)
└── requirements.txt      # Python deps
Caddyfile                 # updated site block below
key.json                  # GCP service‑account key (git‑ignored)
docker-compose.yml        # compose‑only build for translator
.env                      # PROJECT_ID and API_PASSWORD
```

---

## docker-compose.yml additions
*(merge into your existing file — unchanged from the previous step, shown here for reference)*
```yaml
version: "3.9"
services:
  translator:
    image: python:3.11-slim
    container_name: translator
    restart: unless-stopped
    working_dir: /app
    volumes:
      - ./translator:/app
      - ./key.json:/secrets/key.json:ro
    environment:
      PROJECT_ID: ${PROJECT_ID}
      API_PASSWORD: ${API_PASSWORD}
      GOOGLE_APPLICATION_CREDENTIALS: /secrets/key.json
    command: >
      sh -c "apt-get update -qq && \
             apt-get install -y --no-install-recommends libjpeg62-turbo libfreetype6 && \
             pip install --no-cache-dir -r requirements.txt && \
             gunicorn -w 4 -b 0.0.0.0:8080 translate_app:app"
    networks: [internal]
    # Internal only; remove host exposure if you like.
    ports:
      - "127.0.0.1:8080:8080"
```

---

## Caddyfile – new site block
```caddyfile
translate.aaroncollins.info {
    reverse_proxy translator:8080    # No header injection – clients must send X-API-KEY themselves
    encode gzip
}
```
*If you need Basic‑Auth as well, wrap the block with `basicauth` just like your other sub‑domains.*

---

## .env
```
PROJECT_ID=my-gcp-project
API_PASSWORD=super-secret-pass
```

---

### Calling the API (example)
```bash
curl -X POST \
     -H "X-API-KEY: super-secret-pass" \
     -F "file=@page.jpg" \
     -F "target=en" \
     https://translate.aaroncollins.info/translate-image | jq .translated_text
```

The Flask app still exposes `/healthz` without auth.

---

## translator/translate_app.py (unchanged)
[full code preserved below]
"""

import os
import io
import base64
import secrets
from typing import Tuple

from flask import Flask, request, jsonify, abort
from google.cloud import vision
from google.cloud import translate_v3 as translate
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Mandatory configuration – fail fast if missing.
# ---------------------------------------------------------------------------
PROJECT_ID: str = os.getenv("PROJECT_ID", "").strip()
API_PASSWORD: str = os.getenv("API_PASSWORD", "").strip()
LOCATION: str = os.getenv("LOCATION", "global").strip()

if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID env var not set. Export it before running.")
if not API_PASSWORD:
    raise RuntimeError("API_PASSWORD env var not set. Export it before running.")

# Create Cloud clients once per process – thread‑safe.
vision_client = vision.ImageAnnotatorClient()
translate_client = translate.TranslationServiceClient()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Auth guard (401 if key missing or wrong)
# ---------------------------------------------------------------------------

@app.before_request
def enforce_api_key():
    if request.path == "/healthz":  # allow unauthenticated health checks
        return
    client_key = request.headers.get("X-API-KEY") or request.args.get("key")
    if not client_key or not secrets.compare_digest(client_key, API_PASSWORD):
        abort(401, description="Unauthorized – missing or invalid API key")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def translate_text(text: str, target_lang: str) -> str:
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    response = translate_client.translate_text(
        parent=parent,
        contents=[text],
        target_language_code=target_lang,
        mime_type="text/plain",
    )
    return response.translations[0].translated_text


def ocr_image(img_bytes: bytes):
    image = vision.Image(content=img_bytes)
    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        abort(500, f"Vision API error: {response.error.message}")
    return response


def burn_in_translation(img_bytes: bytes, translation: str) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    banner_height = max(40, img.height // 15)
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(0, 0, 0, 180))

    font = ImageFont.load_default()
    draw.text((10, 10), translation, fill=(255, 255, 255, 255), font=font)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/healthz")
def health() -> Tuple[str, int]:
    return "ok", 200


@app.route("/translate-image", methods=["POST"])
def translate_image():
    if "file" not in request.files:
        return jsonify(error="'file' field missing"), 400

    file = request.files["file"]
    target = request.form.get("target", "en")
    img_bytes = file.read()

    # 1) OCR
    ocr_resp = ocr_image(img_bytes)
    full_text = ocr_resp.full_text_annotation.text
    source_lang = (
        ocr_resp.text_annotations[0].locale if ocr_resp.text_annotations else "und"
    )

    # 2) Translate
    translated_text = translate_text(full_text, target)

    # 3) Overlay PNG
    translated_png = burn_in_translation(img_bytes, translated_text)
    b64_png = base64.b64encode(translated_png).decode()

    return jsonify(
        source_language=source_lang,
        target_language=target,
        original_text=full_text,
        translated_text=translated_text,
        translated_image=f"data:image/png;base64,{b64_png}",
    )

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
