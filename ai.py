

"""
ai.py – Lightweight OpenAI helper utilities
===========================================

This file provides the *minimal* subset of helpers referenced by the Streamlit
examples (``encode_image``, ``generate``, ``stream``, ``extract``, and
``generate_title``).  It wraps the OpenAI **Responses** and **Embeddings**
end‑points and adds a few convenience shims so the rest of the project can
import::

    from ai import encode_image, generate, stream, extract, generate_title
"""
from __future__ import annotations

import os
import base64
from io import BytesIO
from typing import Any, Generator, List
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps  # Pillow
from openai import OpenAI

# ------------------------------------------------------------------------------
# 🔑  API client – require OPENAI_API_KEY in the environment
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable missing – "
        "set it before importing `ai` helpers."
    )

ai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------------------------
# 📦  Simple on-disk cache for embeddings
# ------------------------------------------------------------------------------
_CACHE_PATH = Path(__file__).parent / "data" / "embed_cache.json"
try:
    _EMBED_CACHE: dict[str, list[float]] = json.loads(_CACHE_PATH.read_text())
except Exception:
    _EMBED_CACHE = {}

def _save_cache() -> None:
    try:
        _CACHE_PATH.parent.mkdir(exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_EMBED_CACHE))
    except Exception:
        pass

# ------------------------------------------------------------------------------
# 🖼️  Image utilities
# ------------------------------------------------------------------------------


def encode_image(image: Image.Image, max_size: tuple[int, int] = (1024, 1024)) -> str:
    """
    Down‑scale, orient and **base64‑encode** a ``PIL.Image`` so it can be sent
    inline via the OpenAI Responses API.

    Parameters
    ----------
    image
        A `PIL.Image` instance.
    max_size
        Maximum width / height.  Images larger than this are resized *keeping
        aspect ratio.*

    Returns
    -------
    str
        Base64‑encoded ``image/jpeg`` payload ready to embed in a
        ``data:image/jpeg;base64,…`` URI.
    """
    # Correct orientation from EXIF metadata
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Resize if necessary
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size)

    # Encode to JPEG → base64
    buf = BytesIO()
    image.save(buf, format="JPEG", optimize=True, quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------------------
# 🔡  Embeddings helper (sometimes handy standalone)
# ------------------------------------------------------------------------------


def embed(texts: List[str] | str, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Convenience wrapper to fetch OpenAI embeddings.

    Parameters
    ----------
    texts
        Either *one* string or a list of strings.
    model
        Embedding model name (defaults to `"text-embedding-3-small"`).

    Returns
    -------
    numpy.ndarray
        • shape ``(N, D)`` if *texts* is a list  
        • shape ``(D,)``     if *texts* is a single string
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]  # type: ignore[list-item]

    keys: List[str] = []
    missing: List[str] = []
    results: List[np.ndarray | None] = []

    for text in texts:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        keys.append(h)
        cached = _EMBED_CACHE.get(h)
        if cached is None:
            missing.append(text)
            results.append(None)
        else:
            results.append(np.array(cached))

    if missing:
        resp = ai_client.embeddings.create(model=model, input=missing)  # type: ignore[arg-type]
        new_vecs = [np.array(item.embedding) for item in resp.data]
        it = iter(new_vecs)
        for i, res in enumerate(results):
            if res is None:
                vec = next(it)
                results[i] = vec
                _EMBED_CACHE[keys[i]] = vec.tolist()
        _save_cache()

    vecs = np.stack(results)
    return vecs[0] if single else vecs


# ------------------------------------------------------------------------------
# 🧩  Structured content helpers  (works even if `models.chat.Content` missing)
# ------------------------------------------------------------------------------

try:
    from models.chat import Content  # type: ignore
except Exception:  # pragma: no cover – fallback stub
    from dataclasses import dataclass

    @dataclass
    class Content:  # type: ignore[override]
        """
        Minimal stub so `_serialize_content_item` continues to work even if the
        real Pydantic model isn't installed.
        """

        type: str = "text"
        text: str | None = None
        image_url: str | None = None


def _serialize_content_item(part: Any) -> dict[str, Any]:
    """Convert *part* into the schema the Responses API expects."""
    if not isinstance(part, Content):
        # Treat foreign objects as plain text
        return {"type": "input_text", "text": str(part)}

    if part.type in {"text", "input_text"}:
        return {"type": "input_text", "text": part.text or ""}

    if part.type in {"image_url", "input_image"}:
        if not part.image_url:
            raise ValueError("image_url item lacks `image_url` field")
        return {"type": "input_image", "image_url": part.image_url}

    # Fallback
    return {"type": "input_text", "text": part.text or str(part)}


def _chat_message_to_openai(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert local chat‐message dict into OpenAI Responses schema."""
    role = msg.get("role")
    if role not in {"user", "assistant", "system", "tool"}:
        raise ValueError("Chat message must include a valid 'role'")

    content_field = msg.get("content")

    if isinstance(content_field, str):
        # Simple text → wrap
        text_type = "input_text" if role == "user" else "output_text"
        return {"role": role, "content": [{"type": text_type, "text": content_field}]}

    if isinstance(content_field, list):
        converted = [_serialize_content_item(p) for p in content_field]
        return {"role": role, "content": converted}

    raise TypeError(f"Unsupported content type: {type(content_field)}")


# ------------------------------------------------------------------------------
# 🤖  High‑level chat helpers (generate / stream / extract / title)
# ------------------------------------------------------------------------------


def generate(
    request: dict,
    images: List[Image.Image] | None = None,
    model: str = "o4-mini",
) -> str:
    """
    Blocking helper – returns *full* assistant response text.

    * request must have : ``{"messages": […], "system_prompt": "…"}``
    * optionally attach PIL images (sent as extra user messages)
    """
    messages = [_chat_message_to_openai(m) for m in request.get("messages", [])]
    sys_prompt = request.get("system_prompt", "")

    # Attach images
    if images:
        for img in images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encode_image(img)}",
                        }
                    ],
                }
            )

    resp = ai_client.responses.create(
        model=model,
        instructions=sys_prompt,
        input=messages,
        stream=False,
    )
    return getattr(resp, "output_text", "")

def extract(request: dict, return_type):  # return_type is a Pydantic model (class)
    """Return a parsed Pydantic model from the assistant."""
    messages = [_chat_message_to_openai(m) for m in request.get("messages", [])]
    sys_prompt = request.get("system_prompt", "")

    resp = ai_client.responses.parse(
        model="o4-mini",
        instructions=sys_prompt,
        input=messages,
        text_format=return_type,
    )
    return resp.output_parsed
