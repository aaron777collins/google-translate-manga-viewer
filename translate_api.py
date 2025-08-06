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
             gunicorn -w 4 -b 0.0.0.0:9000 translate_api:app"
    networks: [internal]
    # Internal only; remove host exposure if you like.
    ports:
      - "127.0.0.1:9000:9000"
```

---

## Caddyfile – new site block
```caddyfile
translate.aaroncollins.info {
    reverse_proxy translator:9000    # No header injection – clients must send X-API-KEY themselves
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
import textwrap
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


def burn_in_translation(img_bytes: bytes, ocr_annotation, target_lang: str) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Use local bold font bundled with the app
    preferred_font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans-Bold.ttf")
    base_font_size = 24

    # Helper: Try to load the TTF font, fallback if missing/broken
    def load_font(size: int):
        if os.path.isfile(preferred_font_path):
            try:
                return ImageFont.truetype(preferred_font_path, size)
            except Exception as e:
                print(f"Font load failed: {e}")
        return ImageFont.load_default()

    for page in ocr_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                full_text = ""
                bbox_coords = []

                for word in paragraph.words:
                    word_text = "".join([s.text for s in word.symbols])
                    full_text += word_text + " "
                    box_vertices = word.bounding_box.vertices
                    if len(box_vertices) == 4:
                        x0, y0 = box_vertices[0].x, box_vertices[0].y
                        x2, y2 = box_vertices[2].x, box_vertices[2].y
                        # Skip vertical or rotated
                        if abs(y2 - y0) > abs(x2 - x0):
                            continue
                        bbox_coords.append((x0, y0, x2, y2))

                full_text = full_text.strip()
                if not full_text or not bbox_coords:
                    continue

                try:
                    translated = translate_text(full_text, target_lang)
                except Exception:
                    translated = "[translation error]"

                # Get bounding box for paragraph
                min_x = min(x0 for x0, _, _, _ in bbox_coords)
                min_y = min(y0 for _, y0, _, _ in bbox_coords)
                max_x = max(x2 for _, _, x2, _ in bbox_coords)
                max_y = max(y2 for _, _, _, y2 in bbox_coords)

                box_width = max_x - min_x
                box_height = max_y - min_y

                # Dynamically shrink font to fit
                font_size = base_font_size
                while font_size > 8:
                    font = load_font(font_size)
                    wrap_width = max(1, box_width // max(1, font_size // 2))
                    wrapped = textwrap.fill(translated, width=wrap_width)
                    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    if text_w <= box_width and text_h <= box_height:
                        break
                    font_size -= 1
                else:
                    font = load_font(10)
                    wrapped = textwrap.fill(translated, width=box_width // 5)

                # Draw translucent background
                draw.rectangle([(min_x, min_y), (max_x, max_y)], fill=(0, 0, 0, 180))

                # Overlay text
                draw.multiline_text(
                    (min_x + 4, min_y + 2),
                    wrapped,
                    fill=(255, 255, 255, 255),
                    font=font,
                    spacing=2
                )

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

    # 2) Translate full text for JSON return (optional)
    translated_text = translate_text(full_text, target)

    # 3) Translate and burn-in each box
    translated_png = burn_in_translation(img_bytes, ocr_resp.full_text_annotation, target)
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
