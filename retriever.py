from rank_bm25 import BM25Okapi
import numpy as np

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks

        tokenized = [
            chunk.page_content.split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=1):
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[-k:][::-1]

        return [self.chunks[i] for i in top_indices]