import json
from pathlib import Path
from .recall import check_recall_at_five
from .utils import dataset_code_path, dataset_docs_path

def aa():
    p = Path(dataset_code_path)

    with p.open("r") as f:
        x = json.load(f)
    query = ""
    b = 0
    g = 0
    for y in x.values():
        for e in y:
            if check_recall_at_five(e):
                g += 1
            else:
                b += 1


    print("FOUND:", g)
    print("MISSED:", b)