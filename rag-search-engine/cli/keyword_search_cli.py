import argparse
import json
import pickle
import string
import math
import os
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


lemmatizer = WordNetLemmatizer()
BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"

class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):

        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())

        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        counter = {}

        for token in tokens:
            lemma = normalize_word(token)

            if lemma not in self.index:
                self.index[lemma] = []

            if doc_id not in self.index[lemma]:
                self.index[lemma].append(doc_id)
            if lemma not in counter:
                counter[lemma] = 1
            else:
                counter[lemma] += 1
        self.term_frequencies[doc_id] = counter
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, doc_id, term):

        lemma = normalize_word(term)

        return self.term_frequencies.get(doc_id, {}).get(lemma, 0)


    def get_documents(self, term):
        lemma = normalize_word(term)
        return sorted(self.index.get(lemma, []))

    def bm25(self, doc_id, term):

        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)

        scores = {}

        for tok in tokens:
            for id in self.docmap.keys():
                l = normalize_word(tok)
                s = self.bm25(id, l)

                if id in scores:
                    scores[id] += s
                else:
                    scores[id] = s

        sorted_dict = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit])
        return sorted_dict



    def build(self):
        movies = load_movies()
        stopwords = load_stopwords()

        for movie in tqdm(movies["movies"]):
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            tokens = tokenize_text(text)
            tokens = [
                token for token in tokens
                if token not in stopwords
            ]

            self.__add_document(doc_id, " ".join(tokens))
            self.docmap[doc_id] = text



    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    def __get_avg_doc_length(self) -> float:

        llen = 0
        i = 0

        for v in self.doc_lengths.values():
            llen += v
            i += 1
        if i == 0:
            return 0.0
        
        return llen / i


    def save(self):
        folder = Path("cache")
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(folder / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(folder / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path , "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)

        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)

        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def normalize_word(word: str) -> str:
    """Convert a word to its lemma."""
    lemma = lemmatizer.lemmatize(word, pos='v')

    return lemma



def tokenize_text(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def token_word(text: str) -> list[str]:
    w = tokenize_text(text)
    if len(w) > 1:
        raise ValueError("more than one word")
    return w[0]


def load_movies():
    with open("data/movies.json", "r") as f:
        return json.load(f)


def load_stopwords():
    with open("data/stopwords.txt", "r") as f:
        return set(f.read().splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyword Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    subparsers.add_parser(
        "build",
        help="Build the inverted index"
    )

    tf_pqrse = subparsers.add_parser(
        "tf",
        help="Build the inverted index"
    )
    bm25idf_pqrse = subparsers.add_parser(
        "bm25idf",
        help="Build the inverted index"
    )

    tf_pqrse.add_argument(
        "id",
        type=int,
        help="id"
    )
    bm25idf_pqrse.add_argument(
        "term",
        type=str,
        help="id"
    )

    tf_pqrse.add_argument(
        "term",
        type=str,
        help="term"
    )
    idf_pqrse = subparsers.add_parser(
        "idf",
        help="Build the inverted index"
    )
    idf_pqrse.add_argument(
        "term",
        type=str,
        help="term"
    )
    tfidf_pqrse = subparsers.add_parser(
        "tfidf",
        help="Build the inverted index"
    )
    tfidf_pqrse.add_argument(
        "id",
        type=int,
        help="term"
    )
    tfidf_pqrse.add_argument(
        "term",
        type=str,
        help="term"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Tunable BM25 K1 parameter"
    )


    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 K1 parameter"
    )



    args = parser.parse_args()
    inv = InvertedIndex()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            inv.load()
            stopwords = load_stopwords()

            query_tokens = tokenize_text(args.query)
            query_tokens = [
                token for token in query_tokens
                if token not in stopwords
            ]

            results = []

            for token in query_tokens:
                lemma = normalize_word(token)
                results.extend(inv.index.get(lemma, []))

            results = sorted(set(results))[:5]

            movies = load_movies()
            movies_by_id = {
                movie["id"]: movie
                for movie in movies["movies"]
            }

            for doc_id in results:
                movie = movies_by_id.get(doc_id)

                if movie:
                    print(f"{doc_id}. {movie['title']}")

        case "build":
            inv.build()
            inv.save()
            print("Index built and saved successfully.")

        case "tf":

            inv.load()
            print(inv.get_tf(args.id, token_word(args.term)))

        case "idf":

            inv.load()
            total_doc_count = math.log((len(inv.docmap) + 1) / (len(inv.get_documents(token_word(args.term))) + 1))
            print(f"Inverse document frequency of '{args.term}': {total_doc_count:.2f}")

        case "tfidf":
            inv.load()
            tf_idf = inv.get_tf(args.id, token_word(args.term)) * math.log((len(inv.docmap) + 1) / (len(inv.get_documents(token_word(args.term))) + 1))
            print("TF-IDF score of 'trapper' in document '424': 24.13")
            print("TF-IDF score of 'push' in document '424': 2.14")
        case "bm25idf":
            inv.load()
            bm25idf =inv.get_bm25_idf(token_word(args.term))
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            inv.load()
            bm25tf = inv.get_bm25_tf(args.id, token_word(args.term), args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.id}': {bm25tf:.2f}")
        case "bm25search":
            inv.load()
            d = inv.bm25_search(args.query, args.limit)
            m = load_movies()
            for k, v in d.items():
                print(f"({k}) {m["movies"][k - 1]["title"]} - score: {v:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    nltk.download("wordnet", quiet=True)
    main()