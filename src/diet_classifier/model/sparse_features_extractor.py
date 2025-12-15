import hashlib
import json
from typing import Dict, List, Optional, Iterable

class SparseFeatureExtractor:
    """
    Extracts sparse features from tokenized text inputs.
    
    Representation (per token) = [word_one_hot | char_ngram_multi_hot]
    Supports hashing with fixed dimension.
    """

    def __init__(
        self,
        word_dict_size: int = 2000,
        ngram_dict_size: int = 2000,
        ngram_overflow_size: int = 100,
        ngram_min: int = 1,
        ngram_max: int = 3,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        unk_token: str = "[UNK]"
    ):
        self.word_dict_size = word_dict_size
        self.ngram_dict_size = ngram_dict_size
        self.ngram_overflow_size = ngram_overflow_size
        self.total_ngram_dim = ngram_dict_size + ngram_overflow_size
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = [cls_token, unk_token, pad_token]

        # initialize tokenizer
        self.tokenizer = self._whitespace_tokenizer

        # dictionaries
        self.word_dict: Dict[str, int] = {}
        self.ngram_dict: Dict[str, int] = {}

    # =============================
    # Tokenizer
    # =============================
    def _whitespace_tokenizer(self, text: str) -> List[str]:
        text = text.strip()
        if not text.startswith(self.cls_token):
            text = self.cls_token + " " + text
        return text.split()

    # =============================
    # Hash function determinista
    # =============================
    @staticmethod
    def deterministic_hash(s: str) -> int:
        """Returns a positive integer hash for a string using hashlib."""
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h, 16)

    # =============================
    # Char ngrams
    # =============================
    @staticmethod
    def char_ngrams(token: str, n_min: int, n_max: int) -> List[str]:
        """Generate n-grams with word boundaries < and >."""
        token = f"<{token}>"
        ngrams = []
        for n in range(n_min, n_max + 1):
            for i in range(len(token) - n + 1):
                ngrams.append(token[i:i+n])
        return ngrams

    # =============================
    # Build dictionaries
    # =============================
    def build_word_dict(self, corpus: Iterable[str]):
        """Build word dictionary from corpus, limited by word_dict_size."""
        counter: Dict[str, int] = {}
        for text in corpus:
            tokens = self.tokenizer(text)
            for tok in tokens:
                if tok in self.special_tokens:
                    continue
                counter[tok] = counter.get(tok, 0) + 1
        # sort by frequency
        real_word_dict_size = self.word_dict_size - len(self.special_tokens)
        if len(counter) > real_word_dict_size:
            print(f"Warning: vocabulary size {len(counter)} exceeds limit {real_word_dict_size}. Truncating.")
        items = sorted(counter.items(), key=lambda x: -x[1])[:real_word_dict_size]
        self.word_dict = {tok: idx for idx, (tok, _) in enumerate(items)}
        # add special tokens at the end
        special_dict = {tok: (idx+len(self.word_dict)) for idx, tok in enumerate(self.special_tokens)} 
        self.word_dict.update(special_dict)

    def build_ngram_dict(self, corpus: Iterable[str]):
        """Build ngram dictionary from corpus, limited by ngram_dict_size."""
        counter: Dict[str, int] = {}
        for text in corpus:
            tokens = self.tokenizer(text)
            for tok in tokens:
                # skip special tokens (they don't add semantic value)
                if tok in self.special_tokens:
                    continue
                ngrams = self.char_ngrams(tok, self.ngram_min, self.ngram_max)
                for ng in ngrams:
                    counter[ng] = counter.get(ng, 0) + 1
        if len(counter) > self.ngram_dict_size:
            print(f"Warning: ngram vocabulary size {len(counter)} exceeds limit {self.ngram_dict_size}. Truncating.")
        # sort by frequency
        items = sorted(counter.items(), key=lambda x: -x[1])[:self.ngram_dict_size]
        self.ngram_dict = {ng: idx for idx, (ng, _) in enumerate(items)}

    # =============================
    # Save/load dictionaries
    # =============================
    def save_dicts(self, path_word_json: str, path_ngram_json: str):
        with open(path_word_json, "w", encoding="utf-8") as f:
            json.dump(self.word_dict, f, ensure_ascii=False, indent=2)
        with open(path_ngram_json, "w", encoding="utf-8") as f:
            json.dump(self.ngram_dict, f, ensure_ascii=False, indent=2)

    def load_dicts(self, path_word_json: str, path_ngram_json: str):
        with open(path_word_json, "r", encoding="utf-8") as f:
            self.word_dict = json.load(f)
        with open(path_ngram_json, "r", encoding="utf-8") as f:
            self.ngram_dict = json.load(f)

    # =============================
    # Feature extraction
    # =============================
    def token_to_word_index(self, token: str) -> Optional[int]:
        # Return UNK if token not found
        unk_index = self.word_dict.get(self.unk_token)
        return self.word_dict.get(token, unk_index)

    def token_to_ngram_indices(self, token: str) -> List[int]:
        ngrams = self.char_ngrams(token, self.ngram_min, self.ngram_max)
        indices = []
        for ng in ngrams:
            if ng in self.ngram_dict:
                indices.append(self.ngram_dict[ng])
            else:
                # fallback: use deterministic hash and map to overflow area
                idx = self.deterministic_hash(ng) % self.ngram_overflow_size + self.ngram_dict_size
                indices.append(idx)
        return indices

if __name__ == "__main__":
    # Example usage
    corpus = [
        "hola mundo",
        "tortilla de patatas",
        "buenos dias",
        "hola ngram",
        "mundo de ngrams",
        "patatas con jamón y huevo",
        "apagar la luz de la sala"
    ]
    sfe = SparseFeatureExtractor(word_dict_size=20, ngram_dict_size=100, ngram_min=2, ngram_max=4)
    sfe.build_word_dict(corpus)
    sfe.build_ngram_dict(corpus)
    print("Word Dictionary:", sfe.word_dict)
    print("Ngram Dictionary:", sfe.ngram_dict)
    sfe.save_dicts("../data/word_dict.json", "../data/ngram_dict.json")
    text = "tortilla ngram alcachofa"
    tokens = sfe.tokenizer(text)
    for tok in tokens:
        word_idx = sfe.token_to_word_index(tok)
        print(f"Token: {tok}, Word Index: {word_idx}")
        if tok in sfe.special_tokens:
            continue
        ngram_indices = sfe.token_to_ngram_indices(tok)
        print(f"Token: {tok}, Ngram Indices: {ngram_indices}")
