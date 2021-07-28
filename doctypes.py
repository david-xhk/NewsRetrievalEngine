from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Document:
    """Dataclass for a document in the collection."""
    id: int
    title: str
    content: str


@dataclass
class TokenizedDocument(Document):
    """Dataclass for a document that has been tokenized."""
    id: int
    title: list[str] | list[int]  # List of words or word ids
    content: list[str] | list[int]


@dataclass
class LanguageModel:
    """Dataclass for a language model."""
    id: int
    model: dict[str, int]  # Mapping of words to word counts
    total: int
    smoothing_constant: int
