import json
import re


def book_to_sentences(book):
    with open(book, "r") as book:
        raw_text = book.read()
        sentence_boundary_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        sentences = re.split(sentence_boundary_pattern, raw_text)
        cleaned_sentences = clean_sentences(sentences)
        sentences_to_json(cleaned_sentences, book)


def clean_sentences(sentences):
    sentences = [sentence.replace("\n", "") for sentence in sentences]
    sentences = [sentence.replace('"', "") for sentence in sentences]
    sentences = [sentence.replace("�", "") for sentence in sentences]
    sentences = [sentence.replace("", "") for sentence in sentences]
    clean_sentences = [sentence for sentence in sentences if len(sentence) > 10]
    return clean_sentences


def sentences_to_json(sentences):
    with open("data/processed/sentences.json", "w", encoding="utf-8") as sentences_json:
        json.dump(sentences, sentences_json, ensure_ascii=False)

book_to_sentences("data/books/stormofswords.txt")
