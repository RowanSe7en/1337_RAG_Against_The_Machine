import fire
from pathlib import Path
from .indexer import indexer
from .searcher import searcher
from .dataset_searcher import dataset_searcher
from .delete import aa

def index(max_chunk_size, index_folder):
    indexer(max_chunk_size, Path(index_folder))


def search(query, k, index_folder):
    searcher(query, k, Path(index_folder))


def search_dataset(dataset_path, k, save_directory):
    dataset_searcher(Path(dataset_path), k, Path(save_directory))

def answer_dataset(student_search_results_path, k, save_directory):
    ...

def a():
    aa()

if __name__ == "__main__":

    fire.Fire()
