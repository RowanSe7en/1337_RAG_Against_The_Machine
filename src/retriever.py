import re
import pickle
from tqdm import tqdm
from .utils import STOP_WORDS
import math


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

    def bm25_index(self, chunks, index_folder):

        tokenized_chunks = []

        for chunk in tqdm(chunks, desc="Tokenizing chunks", unit="chunk"):
            tokenized_chunks.append(tokenize(chunk.page_content))     

        tokenized_data = index_folder / "tokenized_data.pkl"

        with open(tokenized_data, 'wb') as my_file:
            pickle.dump(tokenized_chunks, my_file)

        print(f"Ingestion complete! Indexed {len(chunks)} chunks under {index_folder}")

    def retrieve(self, query: str, k: int, index_folder):
        
        """Retrieve the most relevant chunks."""

        bm25_index = index_folder / "tokenized_data.pkl"

        with open(bm25_index, "rb") as my_file:
            bm25_retrieved_chunks = pickle.load(my_file)

        tf_list = []
        df_dict = {}
        inverted_index = {}
        all_chunks_len = 0
    
        for idx, chunk in enumerate(bm25_retrieved_chunks, start=0):
            all_chunks_len += len(chunk)
            tf_dict = {"idx": idx, "len": len(chunk)}
            tokens_dict = {}
            for token in chunk:
                if token in tokens_dict:
                    tokens_dict[token] += 1
                else:
                    tokens_dict[token] = 1

            for token in tokens_dict:
                if token in df_dict:
                    df_dict[token] += 1
                else:
                    df_dict[token] = 1

            for token in tokens_dict:
                if token in inverted_index:
                    inverted_index[token][idx] = tokens_dict[token]
                else:
                    inverted_index[token] = {idx: tokens_dict[token]}

            tf_dict["tokens"] = tokens_dict
            tf_list.append(tf_dict)

        query_tokens = tokenize(query)

        idf_scores = {}

        for token in query_tokens:
            if token not in idf_scores and token in df_dict:
                n = len(bm25_retrieved_chunks)
                df = df_dict[token]
                norm = 0.5
                idf = math.log(1 + ((n - df + norm) / (df + norm)))

                idf_scores[token] = idf

        avg_dl = all_chunks_len / len(bm25_retrieved_chunks)

        k1 = 1.5
        b = 0.75

        final_scores = {}

        for token in query_tokens:
            if token not in inverted_index:
                continue

            idf = idf_scores[token]

            for idx, tf in inverted_index[token].items():
                chunk_len = tf_list[idx]["len"]

                tf_normalized = (
                    tf * (k1 + 1)
                ) / (
                    tf + k1 * (1 - b + b * (chunk_len / avg_dl))
                )

                final_scores[idx] = final_scores.get(idx, 0) + idf * tf_normalized

        ranked_chunks = sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        index_file = index_folder / "chunks.pkl"

        with open(index_file, "rb") as my_file:
            chunks = pickle.load(my_file)

        return [
            chunks[idx]
            for idx, _ in ranked_chunks[:k]
        ]
