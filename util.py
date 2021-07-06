import pandas as pd
from doc import Document

def row_to_doc_adapter(row: pd.Series) -> Document:
    return Document(id = row.ID,
                    title = row.title,
                    content = row.content)

def read_docs(docs_path: str, docs_type: str):
    reader = getattr(pd, f'read_{docs_type}')
    df = reader(docs_path)
    docs = df.apply(row_to_doc_adapter, axis=1)
    return docs.tolist()