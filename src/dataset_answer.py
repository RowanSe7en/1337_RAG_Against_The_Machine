from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_models import StudentSearchResultsAndAnswer, MinimalAnswer
import json
from tqdm import tqdm
from .model import ask_model, download_model
from pathlib import Path

def dataset_answer(student_search_results_path, k, save_directory):
    model_path = download_model()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    with open(student_search_results_path, 'r') as my_file:
        retrived = json.load(my_file)

    all_retrived = retrived["search_results"]

    search_results = []

    for retrive in tqdm(all_retrived, desc="Answering datasets", unit="question"):

        query = retrive["question"]
        retrieved_sources = retrive["retrieved_sources"]

        chunks = []

        for i, source in enumerate(retrieved_sources, start=1):
            with open(source["file_path"], 'r') as my_file:
                content = my_file.read()
                chunk_content = content[
                    source["first_character_index"]:source["last_character_index"]
                ]

            chunks.append(f"Chunk {i}:\n{chunk_content}")

        retrieved_chunks = "\n\n".join(chunks)


        answer = ask_model(model, tokenizer, query, retrieved_chunks)

        r = {}

        r["question_id"] = retrive["question_id"]
        r["question"] = query
        r["retrieved_sources"] = retrieved_sources
        r["answer"] = answer

        search_results.append(r)

        MinimalAnswer(**r)
    d = {}
    d["search_results"] = search_results
    d["k"] = k

    StudentSearchResultsAndAnswer(**d)

    save_directory.mkdir(parents=True, exist_ok=True)
    search_results_and_answer_file_path = Path(student_search_results_path).name
    save_file = save_directory / search_results_and_answer_file_path
    with open(save_file, 'w') as my_file:
        json.dump(d, my_file)
        print(f"Saved student_search_results_and_answer to {save_file}")
