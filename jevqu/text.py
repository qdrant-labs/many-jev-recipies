from __future__ import annotations

def state_from_payload(payload: dict, fields: list[str] | None = None, max_chars: int = 1500) -> dict:
    keys = fields or [k for k in payload if not k.startswith("class") and k != "needs_review"]
    out = {}
    for k in keys:
        v = payload.get(k)
        out[k] = v[:max_chars] if isinstance(v, str) else v
    return out
