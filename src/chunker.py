
import math
from pathlib import Path
from .base_models import MinimalSource
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from .semantic_spliter import SemanticSpliter
from .tools import html_separators, shell_separators
from .splitters import py_splitter, separators_splitter, language_splitter

class Chunker(SemanticSpliter):

    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = math.floor(max_chunk_size * 0.3)

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def add_metadata(self, chunks, path):

        for chunk in chunks:
            chunk.metadata["source"] = path

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

    def split_file(self, path: Path) -> list[Document]:
        """Split a file into chunks."""

        with open(path, "r", errors="ignore") as myfile:
            text = myfile.read()

        if path.suffix == ".py":
            chunks = py_splitter(self.max_chunk_size, self.chunk_overlap, text)

        elif path.suffix in {".html", ".htm"}:

            chunks = separators_splitter(self.max_chunk_size, self.chunk_overlap, html_separators, text)

        elif path.suffix in {".sh", ".sample"}:

            chunks = separators_splitter(self.max_chunk_size, self.chunk_overlap, shell_separators, text)

        elif path.suffix == ".js":
            chunks = language_splitter(self.max_chunk_size, self.chunk_overlap, "js", text)

        elif path.suffix in {".h", ".hpp", ".cuh", ".cpp", ".cu", ".inl",}:
            chunks = language_splitter(self.max_chunk_size, self.chunk_overlap, "cpp", text)

        else:
            chunks = self.semantic_split(text,str(path),)

        return self.add_metadata(chunks, str(path))
