from __future__ import annotations

from collections import Counter
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
    title: list[str | int]  # List of words or word ids
    content: list[str | int]


@dataclass
class LanguageModel:
    """Dataclass for a language model."""
    id: int
    counter: Counter[str, int]
    total: int
    smoothing_constant: int
