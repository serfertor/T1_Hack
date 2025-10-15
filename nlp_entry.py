import re
import string
import spacy
from typing import Dict


class QuestionPreprocessor:

    nlp = spacy.load("ru_core_news_md")

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def preprocess(self, text: str) -> Dict[str, str]:
        clean_text = self.clean_text(text)
        doc = self.nlp(clean_text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and (not self.remove_stopwords or not token.is_stop)
        ]
        normalized = " ".join(tokens)
        return {
            "original": text.strip(),
            "clean": clean_text,
            "normalized": normalized,
        }

    def clean_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip()


