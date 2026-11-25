

import json
from .recall import check_recall_at_five
from tqdm import tqdm

def evaluater(student_search_results_path, dataset_path):

    with open(student_search_results_path, 'r') as my_file:
        student_search_results = json.load(my_file)

    with open(dataset_path, 'r') as my_file:
        dataset = json.load(my_file)

    answerd = 0
    total = 0
    search_results = student_search_results["search_results"]
    dataset_result = dataset["rag_questions"]

    for result, data in tqdm(zip(search_results, dataset_result), total=len(search_results), desc="Evaluating Results", unit="Question"):
        if check_recall_at_five(result, data):
            answerd += 1
        total += 1


    print(f"Questions Evaluated {total}")
    print(f"Recall@{len(search_results[0]["retrieved_sources"])} {answerd / total:.3f} ({(answerd / total) * 100:.1f}%)")