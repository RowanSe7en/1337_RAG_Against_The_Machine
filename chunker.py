import math
import re
from pathlib import Path
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
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

    def cut_if_long(self, chunks: list[Document]) -> list[Document]:
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

    def find_chunk_start(self, text: str,
        content: str, search_position: int,
    ) -> int:
        """Find a chunk in the original text."""
        content = content.strip()

        if not content:
            return -1

        start = text.find(content, search_position)
        if start != -1:
            return start

        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        if not lines:
            return -1

        for line in lines[:3]:
            for length in (100, 50, 30, 20, 10):
                if len(line) < length:
                    continue

                start = text.find(line[:length], search_position)

                if start != -1:
                    return start

        return -1

    def add_markdown_offsets(self, text: str,
        chunks: list[Document], path
    ) -> None:
        """Add source offsets to Markdown chunks."""
        search_position = 0

        for chunk in chunks:
            content = chunk.page_content

            start = self.find_chunk_start(
                text,
                content,
                search_position,
            )

            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

            last_line = lines[-1]

            end_start = text.find(
                last_line[:30],
                start,
            )

            if end_start == -1:
                continue

            end = end_start + len(last_line)

            chunk.metadata["start_char"] = start
            chunk.metadata["end_char"] = end

            search_position = end

    def add_html_offsets(
        self,
        text: str,
        chunks: list[Document],
    ) -> None:
        """Add source offsets to HTML chunks."""
        search_position = 0

        for chunk in chunks:
            content = chunk.page_content.strip()

            if not content:
                continue

            first_line = content.splitlines()[0].strip()
            start = text.find(first_line, search_position)

            if start == -1:
                continue

            last_line = content.splitlines()[-1].strip()
            last_start = text.find(last_line, start)

            if last_start == -1:
                continue

            end = last_start + len(last_line)

            chunk.metadata["start_char"] = start
            chunk.metadata["end_char"] = end

            search_position = end

    def split_file(self, path: Path) -> list[Document]:
        """Split a file into chunks."""
        with open(path, "r", errors="ignore") as myfile:
            text = myfile.read()

        if path.suffix == ".md":
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ]
            )

            chunks = splitter.split_text(text)

            self.add_markdown_offsets(text, chunks, path)

            chunks = self.cut_if_long(chunks)

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

            self.add_html_offsets(text, chunks)

        elif path.suffix == ".py":
            splitter = PythonCodeTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )

            chunks = splitter.create_documents([text])

            chunks = self.cut_if_long(chunks)

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