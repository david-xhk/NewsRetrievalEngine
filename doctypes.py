from dataclasses import dataclass, asdict
import json


@dataclass
class Document:
    """Class for a document in the corpus."""
    id: int
    title: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, kvs: dict):
        return cls(**kvs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))


@dataclass
class TokenizedDocument(Document):
    """Class for a document that has been tokenized."""
    id: int
    title: list
    content: list
