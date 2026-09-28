from __future__ import annotations
import hashlib, json, os, time, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from jevqu import JEV_MODEL

MAX_QUESTIONS = 200
API_URL = "https://api.typesafe.ai/v1/systemone"

def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}

def choice(instructions: str, options: dict[str, str | None]) -> dict:
    if len(options) > 251:
        raise ValueError("choice supports at most 250 options plus none")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}

def _check(questions: dict[str, dict]) -> None:
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"{len(questions)} questions > {MAX_QUESTIONS}; split the request")

@dataclass
class Jev:
    api_key: str | None = None
    cache_dir: Path = Path(".jev_cache")
    model: str = JEV_MODEL
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY not set")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]:
        _check(questions)
        body = {"model": self.model, "state": state, "questions": questions}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        delay = 0.5
        for attempt in range(4):
            try:
                answers = self._post(body)["answers"]
                break
            except Exception:  # ponytail: uniform backoff; split 429 vs 5xx if rate limits bite
                if attempt == 3:
                    raise
                time.sleep(delay); delay *= 2
        path.write_text(json.dumps(answers))
        return answers

@dataclass
class FakeJev:
    nouls: dict[str, float] = field(default_factory=dict)
    choices: dict[str, str] = field(default_factory=dict)
    default_noul: float = 0.05
    model: str = JEV_MODEL

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]:
        _check(questions)
        out = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                out[qid] = {"type": "noul", "noul": self.nouls.get(qid, self.default_noul)}
            elif q["type"] == "choice":
                opts = list(q["criteria"]); pick = self.choices.get(qid, opts[-1])
                probs = {o: (0.9 if o == pick else 0.1 / max(1, len(opts) - 1)) for o in opts}
                out[qid] = {"type": "choice", "choice": pick, "confidence": 0.9, "probabilities": probs}
            else:
                raise NotImplementedError(q["type"])
        return out
