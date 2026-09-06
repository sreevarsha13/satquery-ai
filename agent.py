def detect_task(question):
    question = question.lower()

    if "changed" in question or "change" in question:
        return "change_analysis"

    if "optical" in question and "sar" in question:
        return "optical_sar"

    if "where" in question or "highlight" in question or "locate" in question:
        return "grounding"

    if "describe" in question or "description" in question:
        return "captioning"

    return "vqa"