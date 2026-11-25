def check_iou(result_first, result_last, data_start, data_end):
    """Calculate IoU between two character ranges."""
    intersection = max(0, min(result_last, data_end) - max(result_first, data_start))
    union = max(result_last, data_end) - min(result_first, data_start)

    if union == 0:
        return 0.0

    return intersection / union

def check_recall_at_five(result, data):

    retrieved_sources = result["retrieved_sources"]

    if result["question_id"] != data["question_id"]:
        raise ValueError("not same id")

    for source in retrieved_sources:
        if source["file_path"] != data["sources"][0]["file_path"]:
            continue

        result_first = source["first_character_index"]
        result_last = source["last_character_index"]

        data_start = data["sources"][0]["first_character_index"]
        data_end = data["sources"][0]["last_character_index"]

        iou = check_iou(result_first, result_last, data_start, data_end)

        if iou > 0.05:
            return True

    return False