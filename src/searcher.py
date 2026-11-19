from pathlib import Path
from src.retriever import BM25Retriever


def searcher(query, k, index_folder=Path("data/processed/")):

    if not query.strip():
        raise ValueError("query cannot be empty")
    if k <= 0:
        print("falling back to the default top-k = 5")
        k = 5
    bm25Retriever = BM25Retriever()
    best_matches = bm25Retriever.retrieve(query, k, index_folder)

    for bm in best_matches:
        print(f"{bm.metadata['source']} [{bm.metadata['start_char']}:{bm.metadata['end_char']}]")
