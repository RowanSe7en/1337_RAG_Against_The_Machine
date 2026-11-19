from pathlib import Path
from .base_models import UnansweredQuestion
from .retriever import BM25Retriever
from tqdm import tqdm
import json
from .tools import search_results_file_path, processed_path

def dataset_searcher(dataset_path, k, save_directory):

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
            best_matches = bm25Retriever.retrieve(question, k, Path(processed_path))

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
    save_file = save_directory / search_results_file_path
    with open(save_file, 'w') as my_file:
        json.dump(res_dict, my_file)
        print(f"Saved student_search_results to {save_file}")