"""Gemini REST API backend for VLMPlanner.

Uses ``httpx`` to call the Gemini ``generateContent`` REST endpoint directly,
avoiding the ``google-genai`` Python SDK which can deadlock on gRPC channel
initialization in some HPC/container environments.

The REST endpoint used is:
  https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent

Message format translation
--------------------------
OpenAI role → Gemini role:
  ``system``    → ``systemInstruction`` field in the request body
  ``user``      → ``"user"``
  ``assistant`` → ``"model"``

Content block translation:
  ``{"type": "text",      "text": "..."}``  → ``{"text": "..."}`` part
  ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``
                                            → ``{"inlineData": {"mimeType": "...", "data": "..."}}`` part

Rate-limit retry
----------------
Free-tier Gemini endpoints return HTTP 429 when RPM quota is exceeded.
``call_gemini`` retries up to ``max_retries`` times with exponential back-off
capped at 60 s per wait (2 s → 4 s → … → 60 s).  Each retry emits
``logger.warning`` unless ``warn_on_retry=False`` (or
``GEMINI_WARN_ON_RETRY=0``).

An optional ``call_delay`` (seconds) is inserted *before* every API call to
pace requests and reduce RPM pressure on the free tier.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_data_url(data_url: str) -> tuple[str, str]:
    """Return ``(mime_type, base64_data)`` from a ``data:...;base64,...`` URI."""
    header, b64 = data_url.split(",", 1)
    mime = header.split(":")[1].split(";")[0]
    return mime, b64


def _content_to_parts(content: Any) -> list:
    """Convert an OpenAI ``content`` value to a list of Gemini REST part dicts."""
    if isinstance(content, str):
        return [{"text": content}]

    parts = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append({"text": block["text"]})
        elif btype == "image_url":
            url = block["image_url"]["url"]
            mime, b64 = _parse_data_url(url)
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        # Unknown block types silently skipped.
    return parts


def _build_request_body(
    messages: List[Dict[str, Any]],
    max_output_tokens: int,
    temperature: float,
) -> dict:
    """Convert OpenAI message list to a Gemini ``generateContent`` request body."""
    system_texts: list[str] = []
    contents: list = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # Collect plain text from all system messages.
            if isinstance(content, str):
                system_texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        system_texts.append(block["text"])
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts = _content_to_parts(content)

        # Merge consecutive same-role turns (can happen with few-shot examples).
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        },
    }

    if system_texts:
        body["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_texts)}]
        }

    return body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def call_gemini(
    messages: List[Dict[str, Any]],
    model: str,
    max_output_tokens: int,
    temperature: float,
    api_key: str,
    max_retries: int = 8,
    warn_on_retry: bool = True,
    call_delay: float = 0.0,
) -> str:
    """Call the Gemini ``generateContent`` REST endpoint and return raw text.

    Parameters
    ----------
    messages:
        OpenAI-format message list produced by ``PromptManager``.
    model:
        Gemini model name, e.g. ``"gemini-2.5-flash"`` or ``"gemini-2.5-pro"``.
    max_output_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature (0.0 – 2.0).
    api_key:
        Google AI Studio API key (from ``GEMINI_API_KEY`` env var).
    max_retries:
        Retry attempts on HTTP 429 (rate-limit) responses.  Default is 8,
        which provides roughly 4 minutes of cumulative back-off before giving up.
    warn_on_retry:
        Emit ``logger.warning`` on each retry if ``True`` (default).
        Set to ``False`` or ``GEMINI_WARN_ON_RETRY=0`` to suppress.
    call_delay:
        Seconds to sleep *before* the first attempt of every call.  A value of
        2–5 s is sufficient to stay well under the free-tier RPM quota when
        running many planning calls in sequence.

    Returns
    -------
    str
        Raw text from the first candidate part.

    Raises
    ------
    RuntimeError
        On exhausted retries or non-429 HTTP errors.
    VLMRefusalError (from verify2act.pipeline.planner)
        When the response ``finishReason`` indicates a safety/policy block.
    """
    if api_key == "vertex-ai":
        import google.auth
        import google.auth.transport.requests
        creds, project_id = google.auth.default()
        if not project_id:
            project_id = getattr(creds, "quota_project_id", "verify2act")
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        token = creds.token
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model}:generateContent"
        params = {}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    else:
        url = f"{_BASE_URL}/{model}:generateContent"
        params = {"key": api_key}
        headers = {"Content-Type": "application/json"}

    body = _build_request_body(messages, max_output_tokens, temperature)

    last_exc: Optional[Exception] = None

    # Optional inter-call delay to reduce RPM pressure on the free tier.
    if call_delay > 0:
        logger.debug("call_gemini: pre-call delay %.1f s", call_delay)
        time.sleep(call_delay)

    with httpx.Client(timeout=60.0) as client:
        for attempt in range(max_retries):
            try:
                resp = client.post(url, params=params, headers=headers, json=body)

                # ── Rate limit ───────────────────────────────────────────────
                if resp.status_code == 429:
                    is_depleted = False
                    try:
                        err_msg = resp.json().get("error", {}).get("message", "")
                        if "prepayment credits" in err_msg.lower() or "depleted" in err_msg.lower():
                            is_depleted = True
                    except Exception:
                        pass

                    if is_depleted and api_key != "vertex-ai":
                        logger.warning(
                            "AI Studio prepayment credits are depleted. "
                            "Automatically falling back to GCP Vertex AI."
                        )
                        import os
                        os.environ["GEMINI_API_KEY"] = "vertex-ai"
                        api_key = "vertex-ai"
                        
                        import google.auth
                        import google.auth.transport.requests
                        creds, project_id = google.auth.default()
                        if not project_id:
                            project_id = getattr(creds, "quota_project_id", "verify2act")
                        auth_req = google.auth.transport.requests.Request()
                        creds.refresh(auth_req)
                        token = creds.token
                        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model}:generateContent"
                        params = {}
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {token}"
                        }
                        
                        # Retry immediately with Vertex AI
                        resp = client.post(url, params=params, headers=headers, json=body)
                        if resp.status_code == 200:
                            pass
                        elif resp.status_code == 429:
                            wait = min(2 ** (attempt + 1), 60)
                            if attempt < max_retries - 1:
                                if warn_on_retry:
                                    logger.warning(
                                        "Vertex AI rate limit hit (HTTP 429) on attempt %d/%d "
                                        "— retrying in %d s. "
                                        "Response body: %s",
                                        attempt + 1, max_retries, wait, resp.text
                                    )
                                time.sleep(wait)
                                continue
                            raise RuntimeError(
                                f"Vertex AI rate limit exhausted after {max_retries} attempt(s)."
                            )
                        else:
                            raise RuntimeError(
                                f"Vertex AI returned HTTP {resp.status_code}: {resp.text[:300]}"
                            )
                    else:
                        # Capped exponential back-off: 2s, 4s, 8s … up to 60s.
                        wait = min(2 ** (attempt + 1), 60)
                        if attempt < max_retries - 1:
                            if warn_on_retry:
                                logger.warning(
                                    "Gemini rate limit hit (HTTP 429) on attempt %d/%d "
                                    "— retrying in %d s. "
                                    "Response body: %s",
                                    attempt + 1, max_retries, wait, resp.text
                                )
                            time.sleep(wait)
                            continue
                        raise RuntimeError(
                            f"Gemini rate limit exhausted after {max_retries} attempt(s). "
                            "Consider adding --planner-call-delay 3 or switching to a paid-tier model."
                        )

                # ── Other HTTP errors ────────────────────────────────────────
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Gemini API returned HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                data = resp.json()

                # ── Parse candidates ─────────────────────────────────────────
                candidates = data.get("candidates", [])
                if not candidates:
                    pf = data.get("promptFeedback", {})
                    block_reason = pf.get("blockReason", "unknown")
                    from verify2act.pipeline.planner import VLMRefusalError
                    raise VLMRefusalError(
                        f"Gemini returned no candidates "
                        f"(promptFeedback blockReason={block_reason!r}). "
                        "The prompt may have triggered the model's safety filters."
                    )

                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "STOP")

                if finish_reason in ("SAFETY", "RECITATION", "OTHER"):
                    from verify2act.pipeline.planner import VLMRefusalError
                    raise VLMRefusalError(
                        f"Gemini content policy blocked the response "
                        f"(finishReason={finish_reason!r}). "
                        "The prompt may have triggered the model's safety filters."
                    )

                if finish_reason == "MAX_TOKENS":
                    logger.warning(
                        "Gemini response was cut short by the output token limit "
                        "(finishReason=MAX_TOKENS). The JSON response may be truncated. "
                        "Consider increasing --planner-max-tokens (current: %d).",
                        max_output_tokens,
                    )

                # Extract text from the first part of the first candidate.
                parts = candidate.get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()

                if not text:
                    raise RuntimeError(
                        f"Gemini returned an empty response. "
                        f"finishReason={finish_reason}, full response: {json.dumps(data)[:200]}"
                    )

                return text

            except httpx.RequestError as exc:
                raise RuntimeError(f"Gemini HTTP request failed: {exc}") from exc

    raise RuntimeError(
        f"call_gemini: all {max_retries} attempts failed."
    ) from last_exc
