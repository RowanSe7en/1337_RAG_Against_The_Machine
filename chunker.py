import re
import math
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunker:

    def cut_if_long(self, chunks):

        for chunk in chunks:
            added = 0
            if len(chunk.page_content) > 1000:
                added = math.ceil(len(chunk.page_content) / 1000)
                added = math.ceil((len(chunk.page_content) + (added * 40)) / 1000)
                new_chunks = []

                start = 0
                end = 1000

                for i in range(added):
                    new_chunk = Document(
                        page_content=chunk.page_content[start:end],
                        metadata=chunk.metadata.copy(),
                    )

                    new_chunk.metadata["start_char"] = (
                        chunk.metadata["start_char"] + start
                    )
                    new_chunk.metadata["end_char"] = (
                        chunk.metadata["start_char"] + end
                    )

                    new_chunks.append(new_chunk)
                    start = end - 40
                    if i == added - 2:
                        end = len(chunk.page_content)
                    else:
                        end += 1000 - 40
                position = chunks.index(chunk)
                chunks[position:position + 1] = new_chunks
              
    def add_markdown_offsets(self, text: str, chunks):
        headers = list(re.finditer("(?m)^#{1,4} .+$", text))

        search_position = 0

        for chunk in chunks:
            header = None

            if "Header 4" in chunk.metadata:
                header = "#### " + chunk.metadata["Header 4"]
            elif "Header 3" in chunk.metadata:
                header = "### " + chunk.metadata["Header 3"]
            elif "Header 2" in chunk.metadata:
                header = "## " + chunk.metadata["Header 2"]
            elif "Header 1" in chunk.metadata:
                header = "# " + chunk.metadata["Header 1"]

            if header is None:
                continue

            start = text.find(header, search_position)

            if start == -1:
                continue

            end = len(text)

            for match in headers:
                if match.start() > start:
                    end = match.start()
                    break

            chunk.metadata["start_char"] = start
            chunk.metadata["end_char"] = end

            search_position = end

    def add_html_offsets(self, text: str, chunks):

        headers = list(
            re.finditer(
                r"(?is)<h([1-6])[^>]*>.*?</h\1>",
                text,
            )
        )

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

    def split_file(self, path: Path):
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

            self.add_markdown_offsets(text, chunks)
            self.cut_if_long(chunks)

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
                chunk_size=1000,
                chunk_overlap=100,
                add_start_index=True,
            )
            chunks = splitter.create_documents([text])
            self.cut_if_long(chunks)

        elif path.suffix in {".sh", ".sample"}:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
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
                chunk_size=1000,
                chunk_overlap=100,
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
                chunk_size=1000,
                chunk_overlap=100,
                add_start_index=True,
            )
            chunks = splitter.create_documents([text])

        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                add_start_index=True,
            )
            chunks = splitter.create_documents([text])

        for chunk in chunks:
            chunk.metadata["source"] = str(path)
            chunk.metadata["extension"] = path.suffix

            if "start_index" in chunk.metadata:
                start = chunk.metadata.pop("start_index")

                chunk.metadata["start_char"] = start
                chunk.metadata["end_char"] = (
                    start + len(chunk.page_content)
                )

        return chunks


