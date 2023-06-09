import glob
import json
import os
import re

from utils.json_merger import JsonMerger


class PreProcessor:
    def __init__(self):
        self.preprocess_text("data/books/gameofthrones.txt")
        self.preprocess_text("data/books/clashofkings.txt")
        self.preprocess_text("data/books/stormofswords.txt")
        self.preprocess_text("data/books/feastforcrows.txt")
        self.write_json()

    def preprocess_text(self, book):
        with open(book, "r") as book:
            raw_text = book.read()
            sentence_boundary_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
            sentences = re.split(sentence_boundary_pattern, raw_text)
            cleaned_sentences = self.clean_sentences(sentences)
            book_name = self.get_book_name(book)
            self.sentences_to_json(cleaned_sentences, book_name)

    def get_book_name(self, book):
        book_name = str(book.name)
        match = re.search(r"(?<=\/)(\w+)(?=\.)", book_name)
        if match:
            filename = match.group(1)
            return filename

    def sentences_to_json(self, sentences, book_name):
        with open(
            f"data/processed/{book_name}_sentences.json", "w", encoding="utf-8"
        ) as sentences_json:
            json.dump(sentences, sentences_json, ensure_ascii=False)

    def clean_sentences(self, sentences):
        unwanted_chars = ["\n", '"', "�", ""]
        for char in unwanted_chars:
            sentences = [sentence.replace(char, "") for sentence in sentences]
        clean_sentences = [sentence for sentence in sentences if len(sentence) > 10]
        return clean_sentences

    def write_json(self, path="data/processed"):
        files = glob.glob(os.path.join(path, "*.json"))
        merged = []
        for file in files:
            with open(file, "r", encoding="utf-8") as json_file:
                merged.extend(json.load(json_file))
        with open("data/processed/all_books_sentences.json", "w") as merged_json_files:
            json.dump(merged, merged_json_files, ensure_ascii=False)


if __name__ == "__main__":
    preproccesor = PreProcessor()
    json_merger = JsonMerger()
