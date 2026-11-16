"""NFC word-only tokenization; no learned or character fallback vocabulary."""
from collections import Counter
import hashlib
import json
import re
import unicodedata
from pathlib import Path

CONTROLS = ["USER", "BOT", "EOS", "SILENCE", "UNK"]
# Keep case, digits, Unicode letters and internal straight/curly apostrophes.
# English is the training domain; foreign words remain whole words or UNK.
PATTERN = r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]|_"
RULES = {
    "normalization": "NFC; case preserved; whitespace discarded",
    "pattern": PATTERN,
    "vocabulary": "training text only, descending frequency, Unicode lexical ties",
    "unknown": "whole token -> UNK; no character fallback",
    "decode": "single spaces; close punctuation attaches left; open punctuation attaches right; alternating double quotes; controls hidden except UNK",
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class WordTokenizer:
    def __init__(self, vocabulary, frequencies=None, training_sha256=None):
        if (not isinstance(vocabulary, list) or vocabulary[:5] != CONTROLS
                or len(set(vocabulary)) != len(vocabulary) or len(vocabulary) > 8192):
            raise ValueError("invalid word vocabulary")
        self.vocabulary = list(vocabulary)
        self.ids = {word: i for i, word in enumerate(vocabulary)}
        self.frequencies = dict(frequencies or {})
        self.training_sha256 = training_sha256

    @staticmethod
    def split(text):
        return re.findall(PATTERN, unicodedata.normalize("NFC", text))

    @classmethod
    def build(cls, texts, limit=8192):
        if not 5 <= limit <= 8192:
            raise ValueError("vocabulary limit must be 5..8192")
        texts = [unicodedata.normalize("NFC", text) for text in texts]
        counts = Counter(t for text in texts for t in cls.split(text))
        # Corpus occurrences of reserved strings map to their control IDs; reject
        # rather than silently treating natural-language words as control events.
        collision = set(counts) & set(CONTROLS)
        if collision:
            raise ValueError(f"reserved control spelling in corpus: {sorted(collision)}")
        ranked = sorted(counts, key=lambda t: (-counts[t], t))[:limit-5]
        return cls(CONTROLS + ranked, counts, digest(texts))

    def encode(self, text):
        return [self.ids.get(t, 4) for t in self.split(text)]

    def decode(self, ids):
        result = ""
        attach_next = False
        in_quote = False
        for i in ids:
            word = self.vocabulary[int(i)]
            if int(i) < 4:
                continue
            if int(i) == 4:
                word = "<UNK>"
            closing = word in {",", ".", "?", "!", ";", ":", ")", "]", "}", "%", "…", "”"}
            opening = word in {"(", "[", "{", "“", "$", "£"}
            if word == '"':
                closing, opening = in_quote, not in_quote
                in_quote = not in_quote
            result += ("" if not result or closing or attach_next else " ") + word
            attach_next = opening
        return result

    def pairs(self, row):
        q, a = self.encode(row["Q"]), self.encode(row["A"])
        return [0] + q + [1] + a + [2], [3] * (1 + len(q)) + a + [2, 3]

    def metadata(self):
        return {"schema": "dynamath-word/v1", "vocabulary": self.vocabulary,
                "frequencies": self.frequencies, "training_sha256": self.training_sha256,
                "controls": dict(zip(CONTROLS, range(5))), "rules": RULES,
                "size": len(self.vocabulary)}

    @property
    def sha256(self):
        return digest(self.metadata())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.metadata(), ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text())
        if p["schema"] != "dynamath-word/v1" or p["rules"] != RULES:
            raise ValueError("tokenizer rules/schema mismatch")
        return cls(p["vocabulary"], p["frequencies"], p["training_sha256"])
