"""evaluation.py - final scores on the TEST split (attacks run automatically, no prompts).

Run:  python evaluation.py
Meaning-preservation rate comes only from samples a human reviewed with `python attacker.py test`.
"""
import pandas as pd

import config
from attacker import load_goku, prepare_data, load_split, predict, run_attack


def main():
    model = load_goku()
    prepare_data()
    test = load_split("test")
    test["pred"] = predict(model, test["text"].tolist())[0]
    correct = test["pred"] == test["label"]
    clean_acc = correct.mean()
    robust_all = correct.copy()                  # correct AND no enabled attack succeeded
    print(f"Clean accuracy: {clean_acc:.4f} on {len(test)} test sentences")

    reviewed = pd.DataFrame()
    if config.ATTACK_RESULTS_CSV.exists():
        reviewed = pd.read_csv(config.ATTACK_RESULTS_CSV, encoding="utf-8")
        reviewed = reviewed[reviewed["split"] == "test"]

    rows = []
    for attack, on in config.ENABLED_ATTACKS.items():
        if not on:
            continue
        successes, queries = 0, []
        for sid, row in test[correct].iterrows():
            res = run_attack(model, row["text"], int(row["label"]), attack, sid)
            if res and res["success"]:
                successes += 1
                queries.append(res["queries"])
                robust_all[sid] = False
        n_correct = int(correct.sum())
        rev = reviewed[reviewed["attack_type"] == attack] if len(reviewed) else reviewed
        rows.append({
            "attack_type": attack, "total_examples": len(test), "clean_accuracy": clean_acc,
            "original_correct_examples": n_correct, "successful_attacks": successes,
            "attack_success_rate": successes / n_correct,
            "robust_accuracy": (n_correct - successes) / len(test),
            "avg_queries_per_successful_attack": sum(queries) / len(queries) if queries else None,
            "human_reviewed_samples": len(rev),
            "human_meaning_preservation_rate":
                (rev["meaning_preserved"] == "yes").mean() if len(rev) else None})
        print(f"  finished {attack}")

    rows.append({"attack_type": "ANY_ATTACK", "total_examples": len(test),
                 "clean_accuracy": clean_acc, "original_correct_examples": int(correct.sum()),
                 "robust_accuracy": robust_all.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(config.EVAL_SUMMARY_CSV, index=False, encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved to {config.EVAL_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
