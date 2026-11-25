UNSUPPORTED_EXTENSIONS = {
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".ico",
}

STOP_WORDS = {
    "what",
    "what's",
    "how",
    "when",
    "which",
    "can",
    "a",
    "an",
    "the",
    "are",
    "is",
    "does",
    "do",
    "did",
    "was",
    "were",
    "be",
    "been",
    "it",
    "this",
    "that",
    "they",
    "their",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "of",
    "with",
    "by",
    "as",
    "and",
    "or",
    "but",
    "if",
}

html_separators=[
    "\n</html>",
    "\n</body>",
    "\n</div>",
    "\n</section>",
    "\n</article>",
    "\n</header>",
    "\n</footer>",
    "\n</main>",
    "\n</nav>",
    "\n</ul>",
    "\n</ol>",
    "\n</table>",
    "\n</form>",
    "\n</script>",
    "\n</style>",
    "\n\n",
    "\n",
    " ",
    "",
]

shell_separators=[
    "\nfunction ",
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\ncase ",
    "\n\n",
    "\n",
    " ",
    "",
]

max_chunk_size = 1700

processed_path = "data/processed"
corpus_path = "data/raw"

dataset_code_path = "data/datasets/UnansweredQuestions/dataset_code_public.json"
dataset_docs_path = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
search_results_dir_path = "data/output/search_results/UnansweredQuestions"
# search_results_file_path = "StudentSearchResults.json"
# search_results_and_answer_file_path = "StudentSearchResultsAndAnswer.json"

student_search_results_path = "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
search_results_and_answer_path = "data/output/search_results_and_answer/UnansweredQuestions"