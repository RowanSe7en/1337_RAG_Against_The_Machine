from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from .searcher import searcher
from .base_models import StudentSearchResultsAndAnswer
import json
from .tools import search_results_and_answer_file_path
from tqdm import tqdm

def ask_model(model, tokenizer, query: str, retrieved_chunks) -> str:

    prompt = f"""
    Answer the question using ONLY the provided context.

    If the context does not contain the answer, say "I don't know."

    Context:
    {retrieved_chunks}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.15,
    #     pad_token_id=tokenizer.eos_token_id,
    # )

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return answer.strip()


def download_model(model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Download the complete model repository locally."""

    model_path = snapshot_download(repo_id=model_name)

    print(f"Model downloaded to: {model_path}")
    return model_path

def dataset_answer(student_search_results_path, k, save_directory):
    model_path = download_model()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    with open(student_search_results_path, 'r') as my_file:
        retrived = json.load(my_file)

    all_retrived = retrived["search_results"]

    search_results = []

    for retrive in tqdm(all_retrived):

        query = retrive["question"]
        retrieved_sources = retrive["retrieved_sources"]

        chunks = []

        for i, source in enumerate(retrieved_sources, start=1):
            with open(source["file_path"]) as my_file:
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
    d = {}
    d["search_results"] = search_results
    d["k"] = k

    StudentSearchResultsAndAnswer(**d)

    save_directory.mkdir(parents=True, exist_ok=True)
    save_file = save_directory / search_results_and_answer_file_path
    with open(save_file, 'w') as my_file:
        json.dump(search_results, my_file)
        print(f"Saved student_search_results_and_answer to {save_file}")