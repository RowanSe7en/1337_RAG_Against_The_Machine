

import json
from .recall import check_recall_at_five

def evaluater(student_search_results_path, dataset_path):

    with open(student_search_results_path, 'r') as my_file:
        student_search_results = json.load(my_file)

    with open(dataset_path, 'r') as my_file:
        dataset = json.load(my_file)

    answerd = 0
    total = 0
    search_results = student_search_results["search_results"]
    dataset_result = dataset["rag_questions"]

    for result, data in zip(search_results, dataset_result):
        if check_recall_at_five(result, data):
            answerd += 1
        total += 1


    print((answerd / total) * 100)