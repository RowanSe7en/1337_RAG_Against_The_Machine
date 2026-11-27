import pickle
from pathlib import Path
from src.chunker import Chunker
from src.retriever import BM25Retriever
from tqdm import tqdm
from .utils import UNSUPPORTED_EXTENSIONS, corpus_path, processed_path


def indexer(max_chunk_size):

    if max_chunk_size < 10:
        print("falling back to the default max_chunk_size = 1700")
        max_chunk_size = 1700

    all_chunks = []
    all_embeddings = []

    chunker = Chunker(max_chunk_size)
    paths = list(Path(corpus_path).rglob("*"))

    for path in tqdm(
        paths,
        desc="Chunking files",
        unit="file",
    ):
        if ".git" in path.parts:
            continue

        if path.is_file() and path.suffix not in UNSUPPORTED_EXTENSIONS:
            chunks, embeddings = chunker.split_file(path)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

    dire = Path(processed_path)
    dire.mkdir(parents=True, exist_ok=True)

    index_file = dire / "chunks.pkl"
    index_embeddings_file = dire / "embeddings.pkl"

    with open(index_file, "wb") as my_file:
        pickle.dump(all_chunks, my_file)

    with open(index_embeddings_file, "wb") as my_file:
        pickle.dump(all_embeddings, my_file)

    bm25Retriever = BM25Retriever()
    bm25Retriever.bm25_index(all_chunks, dire)