
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)


def py_splitter(max_chunk_size, chunk_overlap, text):

    splitter = PythonCodeTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return splitter.create_documents([text])

def separators_splitter(max_chunk_size, chunk_overlap, separators, text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
    )

    return splitter.create_documents([text])

def language_splitter(max_chunk_size, chunk_overlap, language, text):

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return splitter.create_documents([text])
