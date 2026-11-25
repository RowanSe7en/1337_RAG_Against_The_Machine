import fire
from pathlib import Path
from .indexer import indexer
from .searcher import searcher
from .dataset_searcher import dataset_searcher
from .dataset_answer import dataset_answer
from .responder import responder
from .evaluater import evaluater
from .delete import aa
from .utils import processed_path, max_chunk_size, k, dataset_docs_path, search_results_dir_path, student_search_results_path, answered_dataset_code_path


def index(max_chunk_size=max_chunk_size):
    indexer(max_chunk_size)


def search(query, k=5, index_folder=processed_path):
    best_matches = searcher(query, k, Path(index_folder))

    for bm in best_matches:
        print(f"{bm.metadata['file_path']} [{bm.metadata['first_character_index']}:{bm.metadata['last_character_index']}]")

def answer(query, k=k):
    print(responder(query, k))

def search_dataset(dataset_path=dataset_docs_path, k=k, save_directory=search_results_dir_path):
    dataset_searcher(Path(dataset_path), k, Path(save_directory))

def answer_dataset(student_search_results_path=student_search_results_path, k=5, save_directory=search_results_dir_path):
    dataset_answer(Path(student_search_results_path), k, Path(save_directory))

def evaluate(student_search_results_path=student_search_results_path, dataset_path=answered_dataset_code_path):
    evaluater(Path(student_search_results_path), Path(dataset_path))

def a():
    aa()

if __name__ == "__main__":

    fire.Fire()
