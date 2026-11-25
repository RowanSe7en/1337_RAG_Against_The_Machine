import fire
from pathlib import Path
from .indexer import indexer
from .searcher import searcher
from .dataset_searcher import dataset_searcher
from .dataset_answer import dataset_answer
from .responder import responder
from .evaluater import evaluater
from .delete import aa

def index(max_chunk_size, index_folder):
    indexer(max_chunk_size, Path(index_folder))


def search(query, k, index_folder):
    best_matches = searcher(query, k, Path(index_folder))

    for bm in best_matches:
        print(f"{bm.metadata['file_path']} [{bm.metadata['first_character_index']}:{bm.metadata['last_character_index']}]")

def answer(query, k):
    print(responder(query, k))

def search_dataset(dataset_path, k, save_directory):
    dataset_searcher(Path(dataset_path), k, Path(save_directory))

def answer_dataset(student_search_results_path, k, save_directory):
    dataset_answer(Path(student_search_results_path), k, Path(save_directory))

def evaluate(student_search_results_path, dataset_path):
    evaluater(Path(student_search_results_path), Path(dataset_path))

def a():
    aa()

if __name__ == "__main__":

    fire.Fire()
