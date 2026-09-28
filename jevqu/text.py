from __future__ import annotations

EXCLUDED_KEYS = {"classes", "class_probs", "class_model", "needs_review"}

def state_from_payload(payload: dict, fields: list[str] | None = None, max_chars: int = 1500) -> dict:
    keys = fields or [k for k in payload if k not in EXCLUDED_KEYS]
    out = {}
    for k in keys:
        v = payload.get(k)
        out[k] = v[:max_chars] if isinstance(v, str) else v
    return out
