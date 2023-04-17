import json


def sentences_to_paragraphs(
    sentences_json_path="data/processed/all_books.json",
    paragraphs_json_path="data/processed/all_books_paragraphs.json",
):
    with open(sentences_json_path, "r") as sentences_json:
        sentence_lines = []
        paragraphs = []

        sentences = json.load(sentences_json)
        [sentence_lines.append(sentence) for sentence in sentences]

        for i in range(0, len(sentence_lines), 5):
            paragraph = " ".join(sentence_lines[i : i + 5])
            paragraphs.append(paragraph)

        with open(paragraphs_json_path, "w") as paragraphs_json:
            json.dump(paragraphs, paragraphs_json, ensure_ascii=False)


sentences_to_paragraphs()
