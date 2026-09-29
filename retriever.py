import re
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rank_bm25 import BM25Okapi


STOP_WORDS = {
    "what",
    "what's",
    "how",
    "when",
    "which",
    "can",
    "a",
    "an",
    "the",
    "are",
    "is",
    "does",
    "do",
    "did",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "they",
    "their",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "of",
    "with",
    "by",
    "as",
    "and",
    "or",
    "but",
    "if",
}


def tokenize(text: str) -> list[str]:
    """Tokenize natural language and code identifiers."""
    tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?",
        text.lower(),
    )

    return [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]


class BM25Retriever:

    def bm25_index(self, chunks):

        tokenized_chunks = []

        for chunk in tqdm(chunks, desc="Tokenizing chunks", unit="chunk"):
            tokenized_chunks.append(tokenize(chunk.page_content))
        
        print(f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed/")

        index_folder = Path("data/processed/")
        tokenized_data = index_folder / "tokenized_data.pkl"

        with open(tokenized_data, 'wb') as my_file:
            pickle.dump(tokenized_chunks, my_file)

        print("Building The BM25 index...")

        bm25_object = BM25Okapi(tokenized_chunks)

        bm25_index = index_folder / "bm25_index.pkl"

        with open(bm25_index, 'wb') as my_file:
            pickle.dump(bm25_object, my_file)

        print("The BM25 index built successfully.")

    def retrieve(self, query: str, k: int = 5):
        """Retrieve the most relevant chunks."""

        index_folder = Path("data/processed/")
        bm25_index = index_folder / "bm25_index.pkl"

        with open(bm25_index, "rb") as my_file:
            bm25_object = pickle.load(my_file)

        query_tokens = tokenize(query)

        scores = bm25_object.get_scores(query_tokens)

        top_indices = np.argsort(scores)[-k:][::-1]

        index_folder = Path("data/processed/")

        index_file = index_folder / "chunks.pkl"

        with open(index_file, "rb") as my_file:
            chunks = pickle.load(my_file)

        return [chunks[i] for i in top_indices]