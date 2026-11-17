"""attacker.py - RAVEN attacker with mandatory human review.

  python attacker.py        -> reviews TRAIN split (approved samples can go to training data)
  python attacker.py test   -> reviews TEST split (meaning check only, never used for training)
"""
import csv
import random
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import config
from Model.preprocess import clean_text
from attacks import spelling, code_switching, homoglyph

sys.stdout.reconfigure(encoding="utf-8")   # so Cyrillic characters print on Windows too

ATTACKS = {"spelling": spelling, "code_switching": code_switching, "homoglyph": homoglyph}
ATTACK_NAMES = {"spelling": "Spelling Variation", "code_switching": "Code-Switching",
                "homoglyph": "Homoglyph"}
LABEL_NAMES = {0: "Negative", 1: "Positive"}


# ---------------------------------------------------------------- model
def load_goku():
    for path in (config.SVM_PATH, config.TFIDF_PATH):
        if not path.exists():
            sys.exit(f"ERROR: model file not found: {path}")
    try:
        clf = joblib.load(config.SVM_PATH)
        tfidf = joblib.load(config.TFIDF_PATH)
    except Exception as e:
        sys.exit(f"ERROR: could not load model files (check scikit-learn==1.8.0): {e}")
    if not isinstance(clf, LinearSVC) or not hasattr(tfidf, "vocabulary_"):
        sys.exit("ERROR: svm.pkl must be a LinearSVC and tfidf.pkl a fitted TfidfVectorizer.")
    return clf, tfidf


def predict(model, texts):
    """Uses Goku's own preprocessing + TF-IDF. Returns (labels, decision scores)."""
    clf, tfidf = model
    X = tfidf.transform([clean_text(t) for t in texts])
    return clf.predict(X), clf.decision_function(X)


# ---------------------------------------------------------------- data
def prepare_data():
    """Recreates Goku's exact 80/20 split (same steps as its train.py) once."""
    if (config.TRAIN_CSV.exists() and config.TEST_CSV.exists()
            and config.TRAIN_CSV.stat().st_size > 0 and config.TEST_CSV.stat().st_size > 0):
        return
    if not config.RAW_CSV.exists() or config.RAW_CSV.stat().st_size == 0:
        sys.exit(f"ERROR: dataset not found or empty: {config.RAW_CSV}")
    df = pd.read_csv(config.RAW_CSV, header=None, encoding="utf-8", on_bad_lines="skip")
    if df.shape[1] != 2:
        sys.exit("ERROR: expected 2 columns (sentiment,text) in RomanUrduSentiment.csv")
    df.columns = ["sentiment", "text"]
    df["sentiment"] = df["sentiment"].str.strip()
    df = df[df["sentiment"].isin(["pos", "neg"])]
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    df["label"] = df["sentiment"].map({"pos": 1, "neg": 0})
    df["text"] = df["text"].str.strip()
    train_idx, test_idx = train_test_split(df.index, test_size=config.TEST_SIZE,
                                           random_state=config.SPLIT_SEED, stratify=df["label"])
    df.loc[train_idx, ["text", "label"]].to_csv(config.TRAIN_CSV, index_label="id", encoding="utf-8")
    df.loc[test_idx, ["text", "label"]].to_csv(config.TEST_CSV, index_label="id", encoding="utf-8")
    print(f"Created train.csv ({len(train_idx)}) and test.csv ({len(test_idx)}).")


def load_split(split):
    path = config.TRAIN_CSV if split == "train" else config.TEST_CSV
    df = pd.read_csv(path, index_col="id", encoding="utf-8")
    if not {"text", "label"} <= set(df.columns) or not set(df["label"]) <= {0, 1}:
        sys.exit(f"ERROR: {path} must have text and label (0/1) columns.")
    return df


# ---------------------------------------------------------------- attack search
def run_attack(model, text, true_label, attack, sentence_id):
    """Black-box search: one model query per candidate, stops at the first success.
    query_count = candidate queries only (the original-sentence prediction is not counted).
    If nothing succeeds, returns the candidate closest to the decision boundary."""
    rng = random.Random(f"{config.SEED}-{sentence_id}-{attack}")
    candidates = ATTACKS[attack].generate(text, rng, config.MAX_MODS, config.MAX_CANDIDATES)
    seen, queries, best = {text}, 0, None
    for cand, mods in candidates:
        if cand in seen or clean_text(cand) == "":     # skip duplicates / nothing left to classify
            continue
        seen.add(cand)
        label, score = predict(model, [cand])
        queries += 1
        label, score = int(label[0]), float(score[0])
        result = {"text": cand, "pred": label, "mods": mods, "score": score}
        if label != true_label:
            result.update(success=True, queries=queries)
            return result
        if best is None or abs(score) < abs(best["score"]):
            best = result
    if best:
        best.update(success=False, queries=queries)
    return best


# ---------------------------------------------------------------- terminal review
def ask(prompt, valid):
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            sys.exit("\nInput closed. Progress is saved.")
        if answer == "q":
            sys.exit("\nStopped. Progress is saved; run again to resume.")
        if answer in valid:
            return answer
        print("  Invalid input. Allowed: " + ", ".join(sorted(v or "ENTER" for v in valid)) + " (or q to quit)")


def choose_attacks(available):
    """Lets the user pick which attack to run in this session."""
    print("\nWhich attack do you want to run this session?\n")
    for n, name in enumerate(available, 1):
        print(f"[{n}] {ATTACK_NAMES[name]}")
    print(f"[{len(available) + 1}] All of the above")
    valid = {str(n) for n in range(1, len(available) + 2)}
    pick = int(ask("\nYour choice: ", valid))
    return available if pick == len(available) + 1 else [available[pick - 1]]


def show_panel(text, label, orig_pred, attack, res):
    line = "=" * 50
    print(f"\n{line}\n{'ATTACK RESULT'.center(50)}\n{line}")
    print(f"\nOriginal Sentence:\n{text}")
    print(f"\nGround Truth:\n{LABEL_NAMES[label]}")
    print(f"\nOriginal Prediction:\n{LABEL_NAMES[orig_pred]}")
    print(f"\nAttack Type:\n{ATTACK_NAMES[attack]}")
    print(f"\nChanges Made:\n{res['mods']}")
    print(f"\nAdversarial Sentence:\n{res['text']}")
    print(f"\nAdversarial Prediction:\n{LABEL_NAMES[res['pred']]}")
    print(f"\nAttack Status:\n{'SUCCESSFUL' if res['success'] else 'FAILED'}")
    print(f"\nModel Queries Used:\n{res['queries']}")
    print(f"\n{'-' * 50}\n{'HUMAN REVIEW'.center(50)}\n{'-' * 50}")


def review(split, success):
    """Returns (meaning_preserved, placement)."""
    print("\nDoes the adversarial sentence preserve the original meaning?\n\n[1] Yes\n[2] No")
    meaning = "yes" if ask("\nYour choice: ", {"1", "2"}) == "1" else "no"
    if split == "test":                          # test data is never used for training
        return meaning, "results_only"
    if meaning == "yes":
        print("\nWhere should this sample be stored?\n\n[1] Training Dataset\n"
              "[2] Attack Results Only\n[3] Reject Sample")
        pick = ask("\nYour choice: ", {"1", "2", "3"})
        return meaning, {"1": "training", "2": "results_only", "3": "rejected"}[pick]
    print("\nMeaning changed, so this sample cannot be used for training.\n\n"
          "[1] Attack Results Only\n[2] Reject Sample")
    pick = ask("\nYour choice: ", {"1", "2"})
    return meaning, {"1": "results_only", "2": "rejected"}[pick]


def append_row(path, row):
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def already_reviewed(split):
    if not config.ATTACK_RESULTS_CSV.exists() or config.ATTACK_RESULTS_CSV.stat().st_size == 0:
        return set()
    df = pd.read_csv(config.ATTACK_RESULTS_CSV, encoding="utf-8")
    df = df[df["split"] == split]
    return set(zip(df["sentence_id"], df["attack_type"]))


# ---------------------------------------------------------------- main
def main():
    split = "train"
    if len(sys.argv) > 1:
        if sys.argv[1] != "test":
            sys.exit("Usage: python attacker.py [test]")
        split = "test"
    model = load_goku()
    prepare_data()
    data = load_split(split).sample(frac=1, random_state=config.SEED)   # fixed review order
    done = already_reviewed(split)
    available = [a for a, on in config.ENABLED_ATTACKS.items() if on]
    enabled = choose_attacks(available)
    print(f"\nSplit: {split} | attacks: {', '.join(enabled)} | already reviewed: {len(done)}")
    print("Type q at any prompt to quit (progress is saved after every review).")

    for sid, row in data.iterrows():
        text, label = row["text"], int(row["label"])
        orig_pred = int(predict(model, [text])[0][0])
        if orig_pred != label:               # misclassified originals can never count as attacks
            continue
        for attack in enabled:
            if (sid, attack) in done:
                continue
            res = run_attack(model, text, label, attack, sid)
            if res is None:                  # attack not applicable to this sentence
                continue
            show_panel(text, label, orig_pred, attack, res)
            meaning, placement = review(split, res["success"])
            sample_type = ""
            if placement == "training":
                sample_type = "adversarial" if res["success"] else "augmentation"
                append_row(config.ADV_TRAIN_CSV, {
                    "text": res["text"], "label": label, "sample_type": sample_type,
                    "attack_type": attack, "source_sentence_id": sid})
            append_row(config.ATTACK_RESULTS_CSV, {
                "sentence_id": sid, "split": split, "original_text": text, "true_label": label,
                "original_prediction": orig_pred, "attack_type": attack,
                "adversarial_text": res["text"], "adversarial_prediction": res["pred"],
                "attack_success": res["success"], "query_count": res["queries"],
                "modifications": res["mods"], "meaning_preserved": meaning,
                "placement": placement, "sample_type": sample_type})
            print("\nSample saved successfully.")
            ask("Press ENTER to continue to the next attack...", {""})
    print("\nAll sentences reviewed for the selected attacks.")


if __name__ == "__main__":
    main()
