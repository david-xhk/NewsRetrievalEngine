from dataclasses import dataclass

@dataclass
class Document:
    """Class for a document in the corpus"""
    id: int
    title: str
    content: str
