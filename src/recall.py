from pathlib import Path
from src.retriever import BM25Retriever
from .tools import processed_path


def check_iou(first, last, start, end):
    """Calculate IoU between two character ranges."""
    intersection = max(0, min(last, end) - max(first, start))
    union = max(last, end) - min(first, start)

    if union == 0:
        return 0.0

    return intersection / union

def check_recall_at_five(e, index_folder=Path(processed_path)):
    query = e["question"]

    rag_path = e["sources"][0]["file_path"]

    first = e["sources"][0]["first_character_index"]
    last = e["sources"][0]["last_character_index"]

    bm25Retriever = BM25Retriever()
    best_matches = bm25Retriever.retrieve(query, 5, index_folder)

    for chunk in best_matches:
        if chunk.metadata["source"] != rag_path:
            continue

        start = chunk.metadata["start_char"]
        end = chunk.metadata["end_char"]

        iou = check_iou(first, last, start, end)

        if iou > 0.05:
            return True

    return False