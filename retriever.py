import re

import numpy as np
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
    def __init__(self, chunks):
        self.chunks = chunks

        tokenized_chunks = [
            tokenize(chunk.page_content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query: str, k: int = 5):
        """Retrieve the most relevant chunks."""
        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[-k:][::-1]

        return [self.chunks[i] for i in top_indices]