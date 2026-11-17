"""config.py - all settings in one place."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SVM_PATH = BASE_DIR / "Model" / "svm.pkl"
TFIDF_PATH = BASE_DIR / "Model" / "tfidf.pkl"

DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "RomanUrduSentiment.csv"      # original dataset, never modified
TRAIN_CSV = DATA_DIR / "train.csv"                 # created automatically
TEST_CSV = DATA_DIR / "test.csv"                   # created automatically
ATTACK_RESULTS_CSV = DATA_DIR / "attack_results.csv"
ADV_TRAIN_CSV = DATA_DIR / "adversarial_train.csv"
EVAL_SUMMARY_CSV = DATA_DIR / "evaluation_summary.csv"

# Must match Goku's train.py so the test set is exactly Goku's test set.
SPLIT_SEED = 42
TEST_SIZE = 0.2

# --- Team split: each member reviews a different slice of the data -----------
NUM_MEMBERS = 5           # same on every laptop
MEMBER_ID = 1             # CHANGE PER LAPTOP: 1, 2, 3, 4 or 5

# --- Everything below must be IDENTICAL on every laptop ----------------------
SEED = 42                 # seed for review order and candidate ordering
MAX_MODS = 2              # max modifications per candidate (words or characters)
MAX_CANDIDATES = 30       # max model queries per sentence per attack

ENABLED_ATTACKS = {       # set False to switch an attack off
    "spelling": True,
    "code_switching": True,
    "homoglyph": True,
}

# --- STARTER LISTS: YOU MUST VERIFY / EDIT THESE (lowercase keys) -----------
SPELLING_VARIANTS = {
    "acha": ["achaa", "achha", "accha"],
    "achi": ["achii", "achhi", "acchi"],
    "bohat": ["buhat", "bohot", "bahut"],
    "bohot": ["bohat", "buhat"],
    "bura": ["buraa"],
    "buri": ["burii"],
    "bekar": ["bekaar", "baikar"],
    "kamal": ["kamaal"],
    "zabardast": ["zabardas", "zabardust"],
    "behtareen": ["behtarin", "behtrin"],
    "khubsurat": ["khoobsurat", "khobsurat"],
    "bakwas": ["bakwaas", "bakwass"],
    "pasand": ["pasnd"],
    "theek": ["thik", "theak"],
    "shandar": ["shaandar", "shandaar"],
    "ghalat": ["galat"],
}

CODE_SWITCH_WORDS = {
    "acha": ["good"], "achi": ["good"],
    "bura": ["bad"], "buri": ["bad"],
    "bekar": ["useless"],
    "kamal": ["amazing"],
    "zabardast": ["awesome"],
    "behtareen": ["excellent"],
    "khubsurat": ["beautiful"],
    "bakwas": ["nonsense"],
    "ghalat": ["wrong"],
    "shandar": ["wonderful"],
    "dilchasp": ["interesting"],
    "kahani": ["story"],
}

HOMOGLYPHS = {            # Latin -> Cyrillic look-alikes
    "a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440",
    "c": "\u0441", "i": "\u0456", "y": "\u0443", "x": "\u0445",
}
