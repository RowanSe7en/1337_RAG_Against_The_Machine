
import re
import numpy as np
from langchain_core.documents import Document


class SemanticSpliter():

    def semantic_split(
        self,
        text: str,
        source: str,
    ) -> list[Document]:
        """Split text using embedding-based semantic boundaries."""

        sentence_spans = []

        for match in re.finditer(
            r"\S.*?(?:[.!?](?=\s|$)|\n+|$)",
            text,
            re.DOTALL,
        ):
            start, end = match.span()
            sentence_spans.append((start, end))

        if not sentence_spans:
            return []

        sentences = [
            text[start:end]
            for start, end in sentence_spans
        ]

        embeddings = self.embedding_model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        distances = 1 - np.sum(
            embeddings[:-1] * embeddings[1:],
            axis=1,
        )

        threshold = (
            np.percentile(distances, 95)
            if len(distances) > 0
            else float("inf")
        )

        chunks = []
        current_start = 0
        current_end = 0

        def save_chunk(start_index, end_index):
            start_char = sentence_spans[start_index][0]
            end_char = sentence_spans[end_index][1]

            content = text[start_char:end_char]

            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                )
            )

        for i, (start, end) in enumerate(sentence_spans):

            if i == 0:
                current_start = 0
                current_end = 0
                continue

            next_size = (
                end - sentence_spans[current_start][0]
            )

            semantic_boundary = distances[i - 1] > threshold

            size_boundary = next_size > self.max_chunk_size

            if semantic_boundary or size_boundary:
                save_chunk(current_start, current_end)

                overlap_start = current_end + 1

                while overlap_start > current_start:
                    overlap_size = (
                        sentence_spans[current_end][1]
                        - sentence_spans[overlap_start - 1][0]
                    )

                    if overlap_size > self.chunk_overlap:
                        break

                    overlap_start -= 1

                current_start = overlap_start
                current_end = i

            else:
                current_end = i

        save_chunk(current_start, current_end)

        return chunks