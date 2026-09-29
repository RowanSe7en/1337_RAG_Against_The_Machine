import json
import fire
import pickle
from pathlib import Path
from chunker import Chunker
from retriever import BM25Retriever
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import List


UNSUPPORTED_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".ico",
}

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List
    answer: str

def check_iou(first, last, start, end):
    """Calculate IoU between two character ranges."""
    intersection = max(0, min(last, end) - max(first, start))
    union = max(last, end) - min(first, start)

    if union == 0:
        return 0.0

    return intersection / union


def check_recal_at_five(e):
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

    if not query.strip():
        raise ValueError("query cannot be empty")
    if k <= 0:
        print("falling back to the default top-k = 5")
        k = 5
    bm25Retriever = BM25Retriever()
    best_matches = bm25Retriever.retrieve(query, k)

    for bm in best_matches:
        print(f"{bm.metadata['source']} [{bm.metadata['start_char']}:{bm.metadata['end_char']}]")


def index(max_chunk_size=1700, folder_path=Path("data/processed/")):

    if max_chunk_size < 10:
        print("falling back to the default max_chunk_size = 1700")
        max_chunk_size = 1700


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

    index_folder = folder_path
    index_folder.mkdir(parents=True, exist_ok=True)

    index_file = index_folder / "chunks.pkl"

    with open(index_file, "wb") as my_file:
        pickle.dump(all_chunks, my_file)

    bm25Retriever = BM25Retriever()
    bm25Retriever.bm25_index(all_chunks)


def search_dataset(dataset_path=Path("data/datasets/UnansweredQuestions/dataset_code_public.json"), k=5, save_directory=Path("data/output/search_results/UnansweredQuestions")):

    if k <= 0:
        print("falling back to the default top-k = 5")
        k = 5

    with open(dataset_path, 'r') as my_file:
        questions = json.load(my_file)

    res_dict = {"search_results": [], "k": k}

    for v in questions.values():
        for e in tqdm(list(v), desc="Searching datasets", unit="question"):
            answer_dict = {}
            UnansweredQuestion(**e)
            question_id = e['question_id']
            question = e['question']

            bm25Retriever = BM25Retriever()
            best_matches = bm25Retriever.retrieve(question, k)

            answer_dict['question'] = question
            answer_dict['question_id'] = question_id
            answer_dict['retrieved_sources'] = []

            for bm in best_matches:
                file_dict = {}
                file_dict['file_path'] = bm.metadata['source']
                file_dict['first_character_index'] = bm.metadata['start_char']
                file_dict['last_character_index'] = bm.metadata['end_char']
                answer_dict['retrieved_sources'].append(file_dict)
            res_dict['search_results'].append(answer_dict)

    save_directory.mkdir(parents=True, exist_ok=True)
    save_file = save_directory / "dataset_docs_public.json"
    with open(save_file, 'w') as my_file:
        json.dump(res_dict, my_file)
        print(f"Saved student_search_results to {save_file}")

def answer_dataset(student_search_results_path=Path(" data/output/search_results/UnansweredQuestions/dataset_docs_public.json"), k=5, save_directory=Path("data/output/search_results_and_answer/UnansweredQuestions")):
    ...

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
#             if check_recal_at_five(e):
#                 g += 1
#             else:
#                 b += 1


# print("FOUND:", g)
# print("MISSED:", b)