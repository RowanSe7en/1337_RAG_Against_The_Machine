
import math
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunker:

    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = math.floor(max_chunk_size * 0.3)

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

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

            # while start < end and text[start].isspace():
            #     start += 1

            # while end > start and text[end - 1].isspace():
            #     end -= 1

            # if start < end:
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

                # Preserve sentence-level overlap.
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

    def cut_if_long(
        self,
        chunks: list[Document],
    ) -> list[Document]:
        """Cut chunks that exceed max_chunk_size."""

        result = []

        for chunk in chunks:
            if len(chunk.page_content) <= self.max_chunk_size:
                result.append(chunk)
                continue

            source_start = chunk.metadata["start_char"]
            content_length = len(chunk.page_content)

            start = 0

            while start < content_length:
                end = min(
                    start + self.max_chunk_size,
                    content_length,
                )

                new_chunk = Document(
                    page_content=chunk.page_content[start:end],
                    metadata=chunk.metadata.copy(),
                )

                new_chunk.metadata["start_char"] = (
                    source_start + start
                )
                new_chunk.metadata["end_char"] = (
                    source_start + end
                )

                result.append(new_chunk)

                if end == content_length:
                    break

                start = end - self.chunk_overlap

        return result

    def find_chunk_start(
        self,
        text: str,
        content: str,
        search_position: int,
    ) -> int:
        """Find a chunk in the original text."""

        start = text.find(content, search_position)

        if start != -1:
            return start

        lines = []
        offset = 0

        for line in content.splitlines(keepends=True):
            lines.append((line, offset))
            offset += len(line)

        if not lines:
            return -1

        for line, line_offset in lines[:3]:
            for length in (100, 50, 30, 20, 10):
                if len(line) < length:
                    continue

                start = text.find(
                    line[:length],
                    search_position,
                )

                if start != -1:
                    return start - line_offset

        return -1

    def add_offsets(
        self,
        text: str,
        chunks: list[Document],
        path=None,
    ) -> None:
        """Add source offsets to Markdown or HTML chunks."""

        search_position = 0

        for chunk in chunks:
            content = chunk.page_content

            if not content:
                continue

            lines = [
                line
                for line in content.splitlines()
                if line
            ]

            if not lines:
                continue

            start = text.find(
                lines[0][:30],
                search_position,
            )

            if start == -1:
                raise ValueError("couldn't find start index")

            last_line = lines[-1]

            end_start = text.find(
                last_line[:30],
                start,
            )

            if end_start == -1:
                raise ValueError("couldn't find end index")

            end = end_start + len(last_line)

            chunk.metadata["start_char"] = start
            chunk.metadata["end_char"] = end

            search_position = end

    def split_file(self, path: Path) -> list[Document]:
        """Split a file into chunks."""

        with open(path, "r", errors="ignore") as myfile:
            text = myfile.read()

        if path.suffix in {".md", ".txt", ".rst"}:
            chunks = self.semantic_split(
                text,
                str(path),
            )

        elif path.suffix == ".html":
            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                    ("h5", "Header 5"),
                    ("h6", "Header 6"),
                ]
            )

            chunks = splitter.split_text(text)

            self.add_offsets(text, chunks, path)
            chunks = self.cut_if_long(chunks)

        elif path.suffix == ".py":
            splitter = PythonCodeTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix in {".sh", ".sample"}:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\nfunction ",
                    "\nif ",
                    "\nfor ",
                    "\nwhile ",
                    "\ncase ",
                    "\n\n",
                    "\n",
                    " ",
                    "",
                ],
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix == ".js":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language="js",
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix in {
            ".h",
            ".hpp",
            ".cuh",
            ".cpp",
            ".cu",
            ".inl",
        }:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language="cpp",
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

        for chunk in chunks:
            chunk.metadata["source"] = str(path)

            if "start_index" in chunk.metadata:
                start = chunk.metadata.pop("start_index")

                chunk.metadata["start_char"] = start
                chunk.metadata["end_char"] = (
                    start + len(chunk.page_content)
                )

            MinimalSource(
                file_path=chunk.metadata["source"],
                first_character_index=chunk.metadata["start_char"],
                last_character_index=chunk.metadata["end_char"],
            )

        return chunks
