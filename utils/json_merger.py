import json


class JsonMerger:
    """
    Use this class if you want to merge sentences from your json file
    of sentences into paragraphs of a specific length for longer training inputs.
    """

    def __init__(
        self,
        sentences_json_path="data/processed/all_books.json",
        paragraphs_json_path="data/processed/all_books_paragraphs.json",
        paragraph_length=5,
    ):
        self.sentences_json_path = sentences_json_path
        self.paragraphs_json_path = paragraphs_json_path
        self.paragraph_length = paragraph_length
        self.sentences_to_paragraphs()

    def sentences_to_paragraphs(self):
        with open(self.sentences_json_path, "r") as sentences_json:
            sentence_lines = []
            paragraphs = []

            sentences = json.load(sentences_json)
            [sentence_lines.append(sentence) for sentence in sentences]

            for i in range(0, len(sentence_lines), self.paragraph_length):
                paragraph = " ".join(sentence_lines[i : i + self.paragraph_length])
                paragraphs.append(paragraph)

            with open(self.paragraphs_json_path, "w") as paragraphs_json:
                json.dump(paragraphs, paragraphs_json, ensure_ascii=False)


if __name__ == "__main__":
    json_merger = JsonMerger()
