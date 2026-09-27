from pathlib import Path
from chunker import Chunker
from retriever import BM25Retriever
import json

UNSUPPORTED_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".ico",
}

def check_iou(first, last, start, end):
    """Calculate IoU between two character ranges."""
    intersection = max(0, min(last, end) - max(first, start))
    union = max(last, end) - min(first, start)

    if union == 0:
        return 0.0

    return intersection / union


def check_source(e):
    query = e["question"]

    rag_path = e["sources"][0]["file_path"]

    first = e["sources"][0]["first_character_index"]
    last = e["sources"][0]["last_character_index"]

    bm = bm25Retriever.retrieve(query)

    for chunk in bm:
        if chunk.metadata["source"] != rag_path:
            continue

        start = chunk.metadata["start_char"]
        end = chunk.metadata["end_char"]

        iou = check_iou(first, last, start, end)

        if iou > 0.05:
            return True

        source_chunks = [
            chunk
            for chunk in bm
            if chunk.metadata["source"] == rag_path
        ]

        for size in range(2, len(source_chunks) + 1):
            for i in range(len(source_chunks) - size + 1):
                combined = source_chunks[i:i + size]

                start = min(
                    chunk.metadata["start_char"]
                    for chunk in combined
                )

                end = max(
                    chunk.metadata["end_char"]
                    for chunk in combined
                )

                iou = check_iou(first, last, start, end)

                if iou > 0.05:
                    return True

    return False

if __name__ == "__main__":

    all_chunks = []

    chunker = Chunker(1700)

    for path in Path("./data/raw").rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file() and path.suffix not in UNSUPPORTED_EXTENSIONS:
            new = chunker.split_file(path)
            all_chunks.extend(new)

    bm25Retriever = BM25Retriever(all_chunks)
    p = Path("datasets_public/public/AnsweredQuestions/dataset_docs_public.json")

    with p.open("r") as f:
        x = json.load(f)
    query = ""
    b = 0
    g = 0
    for k, y in x.items():
        for e in y:
            if check_source(e):
                g += 1
            else:
                b += 1


print("FOUND:", g)
print("MISSED:", b)