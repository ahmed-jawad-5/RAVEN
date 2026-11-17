"""Attack 3: replace Latin letters with look-alike Cyrillic letters."""
import re
from attacks import pick_combinations
from config import HOMOGLYPHS
from Model.preprocess import STOPWORDS


def generate(text, rng, max_mods, limit):
    parts = re.split(r"(\w+)", text)
    slots = {}                                 # (word index, char index) -> [look-alike]
    for i, word in enumerate(parts):
        if i % 2 == 1 and word.lower() not in STOPWORDS:   # stopwords are removed anyway
            for j, ch in enumerate(word):
                if ch in HOMOGLYPHS:
                    slots[(i, j)] = [HOMOGLYPHS[ch]]
    result = []
    for combo in pick_combinations(slots, max_mods, limit, rng):
        new = parts.copy()
        for (i, j), glyph in combo:
            chars = list(new[i])
            chars[j] = glyph
            new[i] = "".join(chars)
        result.append(("".join(new), ", ".join(f"{parts[i][j]}->{g}" for (i, j), g in combo)))
    return result
