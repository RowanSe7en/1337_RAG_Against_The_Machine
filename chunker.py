from pathlib import Path

from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunker:

    def split_file(self, path: Path):

        with open(path, "r", errors="ignore") as myfile:
            text = myfile.read()

        if path.suffix == ".md":

            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )

            chunks = splitter.split_text(text)

        elif path.suffix == ".py":


            splitter = PythonCodeTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix in {".sh", ".sample"}:


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=[
                    "\nfunction ", "\nif ", "\nfor ", "\nwhile ",
                     "\ncase ", "\n\n", "\n", " ", ""
                    ]
            )

            chunks = splitter.create_documents([text])

        elif path.suffix == ".js":


            splitter = RecursiveCharacterTextSplitter.from_languague(
                extension=path.suffix,
                chunk_size=1000,
                chunk_overlap=100,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix in {".h", ".hpp", ".cuh", ".cpp", ".cu", ".inl"}:

            splitter = RecursiveCharacterTextSplitter.from_extension(
                extension=path.suffix,
                chunk_size=1000,
                chunk_overlap=100,
            )

            chunks = splitter.create_documents([text])

        elif path.suffix == ".html":

            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )

            chunks = splitter.split_text(text)

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            )

            chunks = splitter.create_documents([text])

        for chunk in chunks:

            chunk.metadata["source"] = str(path)
            chunk.metadata["extension"] = path.suffix

        return chunks


all_chunks = []

chunker = Chunker()

for path in Path(".").rglob("*"):
    if path.is_file():
        all_chunks.extend(chunker.split_file(path))

# print(all_chunks[0].metadata)
