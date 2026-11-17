"""Attack 2: replace words with verified English substitutes from config."""
import re
from attacks import pick_combinations
from config import CODE_SWITCH_WORDS
from Model.preprocess import STOPWORDS


def generate(text, rng, max_mods, limit):
    parts = re.split(r"(\w+)", text)
    slots = {i: CODE_SWITCH_WORDS[w.lower()] for i, w in enumerate(parts)
             if i % 2 == 1 and w.lower() in CODE_SWITCH_WORDS and w.lower() not in STOPWORDS}
    result = []
    for combo in pick_combinations(slots, max_mods, limit, rng):
        new = parts.copy()
        for i, english in combo:
            new[i] = english
        result.append(("".join(new), ", ".join(f"{parts[i]}->{e}" for i, e in combo)))
    return result
