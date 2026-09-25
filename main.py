from pathlib import Path
from chunker import Chunker
from retriever import BM25Retriever

UNSUPPORTED_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".ico",
}

if __name__ == "__main__":

    all_chunks = []

    chunker = Chunker()

    x = 0

    for path in Path(".").rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file() and path.suffix not in UNSUPPORTED_EXTENSIONS:
            new = chunker.split_file(path)
            all_chunks.extend(new)

    bm25Retriever = BM25Retriever(all_chunks)
    query = "what are the UNSUPPORTED_EXTENSIONS"
    bm = bm25Retriever.retrieve(query)
    # for e in bm:
    #     print(e)
    #     print("---------------------------------------")