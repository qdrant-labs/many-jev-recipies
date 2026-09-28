from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field

@dataclass
class Facet:
    field: str
    values: list[str]

@dataclass
class ClassDef:
    id: str
    name: str
    definition: str
    exclusions: str
    parent: str | None = None

@dataclass
class Thresholds:
    filter_above: float = 0.9
    boost_above: float = 0.6
    label_above: float = 0.5
    boost_scale: float = 0.01

@dataclass
class Taxonomy:
    version: str
    model: str
    classes: list[ClassDef]
    facets: list[Facet]
    thresholds: Thresholds = field(default_factory=Thresholds)

    def level1(self) -> list[ClassDef]:
        return [c for c in self.classes if c.parent is None]

    def children(self, parent_id: str) -> list[ClassDef]:
        return [c for c in self.classes if c.parent == parent_id]

    def by_id(self, cid: str) -> ClassDef:
        return next(c for c in self.classes if c.id == cid)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, s: str) -> "Taxonomy":
        d = json.loads(s)
        return cls(version=d["version"], model=d["model"],
                   classes=[ClassDef(**c) for c in d["classes"]],
                   facets=[Facet(**f) for f in d["facets"]],
                   thresholds=Thresholds(**d["thresholds"]))

@dataclass
class Understanding:
    classes: list[tuple[str, float]]
    facets: dict[str, tuple[str, float]]
    taxonomy_version: str
