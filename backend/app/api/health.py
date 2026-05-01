"""LLM-Verfügbarkeits-Ampel: cached Mini-Test-Call gegen apigen."""

import time
from datetime import datetime, timezone

from flask import jsonify
from openai import APIError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

from ..config import Config
from . import health_bp

_CACHE_TTL_SECONDS = 1800  # 30 Min — kostet ~48 Test-Calls/Tag bei Dauerlast
_cache = {"ts": 0.0, "payload": None}


def _build_payload(status: str, reason: str, http_code: int | None = None) -> dict:
    return {
        "status": status,
        "reason": reason,
        "http_code": http_code,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _probe_llm() -> dict:
    if not Config.LLM_API_KEY or not Config.LLM_BASE_URL:
        return _build_payload("yellow", "config_missing")

    client = OpenAI(api_key=Config.LLM_API_KEY, base_url=Config.LLM_BASE_URL, timeout=8.0)
    try:
        client.chat.completions.create(
            model=Config.LLM_MODEL_NAME or "smart",
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
        return _build_payload("green", "ok", 200)
    except RateLimitError:
        return _build_payload("red", "quota_exceeded", 429)
    except AuthenticationError:
        return _build_payload("yellow", "auth_failed", 401)
    except BadRequestError as exc:
        return _build_payload("yellow", "bad_request", getattr(exc, "status_code", 400))
    except APIError as exc:
        return _build_payload("yellow", "api_error", getattr(exc, "status_code", None))
    except Exception:
        return _build_payload("yellow", "unreachable")


@health_bp.route("/llm", methods=["GET"])
def llm_status():
    now = time.time()
    if _cache["payload"] and (now - _cache["ts"]) < _CACHE_TTL_SECONDS:
        return jsonify({**_cache["payload"], "cached": True})

    payload = _probe_llm()
    _cache["ts"] = now
    _cache["payload"] = payload
    return jsonify({**payload, "cached": False})
