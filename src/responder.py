from transformers import AutoTokenizer, AutoModelForCausalLM
from .searcher import searcher
from .utils import processed_path
from pathlib import Path
from .model import ask_model, download_model

def responder(query, k):

    model_path = download_model()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    retrieved_sources = searcher(query, k, Path(processed_path))

    chunks = []

    for i, source in enumerate(retrieved_sources, start=1):
        print("----------------------------------")
        print(source)
        print("----------------------------------")

        with open(source.metadata["file_path"], 'r') as my_file:
            content = my_file.read()
            chunk_content = content[
                source.metadata["first_character_index"]:source.metadata["last_character_index"]
            ]

        chunks.append(f"Chunk {i}:\n{chunk_content}")

    retrieved_chunks = "\n\n".join(chunks)


    answer = ask_model(model, tokenizer, query, retrieved_chunks)

    return answer