"""Attack 1: spelling variations from a configurable list of known variants."""
import re
from attacks import pick_combinations
from config import SPELLING_VARIANTS
from Model.preprocess import STOPWORDS


def generate(text, rng, max_mods, limit):
    parts = re.split(r"(\w+)", text)          # odd indexes are words
    slots = {i: SPELLING_VARIANTS[w.lower()] for i, w in enumerate(parts)
             if i % 2 == 1 and w.lower() in SPELLING_VARIANTS and w.lower() not in STOPWORDS}
    result = []
    for combo in pick_combinations(slots, max_mods, limit, rng):
        new = parts.copy()
        for i, variant in combo:
            new[i] = variant
        result.append(("".join(new), ", ".join(f"{parts[i]}->{v}" for i, v in combo)))
    return result
