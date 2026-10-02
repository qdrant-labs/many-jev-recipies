from __future__ import annotations
import hashlib, json, os, tempfile, time, urllib.error, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"

@dataclass
class LLM:
    """A cheap chat model that writes class names and definitions during induction: Jev chooses among options, it
    cannot write them. Answers are JSON objects, cached on disk like Jev's."""
    model: str = "qwen/qwen3.8-flash"
    api_key: str | None = None
    cache_dir: Path = Path(".llm_cache")
    timeout: float = 120.0
    usage: list = field(default_factory=list, repr=False)  # USD per uncached request, from the response

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def ask_json(self, system: str, user: str) -> dict:
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "reasoning": {"enabled": False}, "usage": {"include": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        path = self.cache_dir / (hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() + ".json")
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(6):
            try:
                resp = self._post(body)
                break
            except urllib.error.HTTPError as e:  # providers rate-limit bursts (429); other 4xx are our fault
                if (400 <= e.code < 500 and e.code != 429) or attempt == 5:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
        self.usage.append(float((resp.get("usage") or {}).get("cost") or 0.0))
        text = resp["choices"][0]["message"]["content"]
        out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        with tempfile.NamedTemporaryFile("w", dir=self.cache_dir, delete=False, suffix=".tmp") as tmp:
            tmp.write(json.dumps(out))
        os.replace(tmp.name, path)
        return out
