import json
import fire
import pickle
from pathlib import Path
from chunker import Chunker
from retriever import BM25Retriever
from tqdm import tqdm

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

    return False

def search(query, k=5):
    bm25Retriever = BM25Retriever()
    best_matches = bm25Retriever.retrieve(query, k)

    for bm in best_matches:
        print(f"{bm.metadata['source']} [{bm.metadata['start_char']}:{bm.metadata['end_char']}]")


def index(max_chunk_size=1700):

    all_chunks = []

    chunker = Chunker(max_chunk_size)
    paths = list(Path("./data/raw").rglob("*"))
    for path in tqdm(
        paths,
        desc="Chunking files",
        unit="file",
    ):
        if ".git" in path.parts:
            continue

        if path.is_file() and path.suffix not in UNSUPPORTED_EXTENSIONS:
            all_chunks.extend(chunker.split_file(path))

    index_folder = Path("data/processed/")
    index_folder.mkdir(parents=True, exist_ok=True)

    index_file = index_folder / "chunks.pkl"

    with open(index_file, "wb") as my_file:
        pickle.dump(all_chunks, my_file)

    bm25Retriever = BM25Retriever()
    bm25Retriever.bm25_index(all_chunks)
    

if __name__ == "__main__":

    fire.Fire()

#     p = Path("datasets_public/public/AnsweredQuestions/dataset_docs_public.json")

#     with p.open("r") as f:
#         x = json.load(f)
#     query = ""
#     b = 0
#     g = 0
#     for k, y in x.items():
#         for e in y:
#             if check_source(e):
#                 g += 1
#             else:
#                 b += 1


# print("FOUND:", g)
# print("MISSED:", b)