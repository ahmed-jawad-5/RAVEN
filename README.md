# RAVEN: Roman Urdu Adversarial Attacker

RAVEN attacks an existing Roman Urdu sentiment classifier (a LinearSVC with TF-IDF, called Goku) with
three kinds of text changes. A human then reviews every generated sentence in the terminal.

| Attack | What it does |
|---|---|
| Spelling Variation | Swaps words for known Roman Urdu spelling variants (bohat -> buhat) |
| Code-Switching | Swaps words for English equivalents (acha -> good) |
| Homoglyph | Replaces Latin letters with look-alike Cyrillic letters |

Nothing is added to the training data unless a human approves it.

## 1. Requirements

- **Python 3.12** (use this exact version so everyone gets identical results)
- scikit-learn **1.8.0** (the saved model was built with it; other versions can give wrong results)

## 2. Project layout

```
RAVEN/
├── Model/        svm.pkl, tfidf.pkl, preprocess.py   (Goku's files, do not edit)
├── attacks/      spelling.py, code_switching.py, homoglyph.py
├── data/         RomanUrduSentiment.csv              (original dataset, never modified)
├── attacker.py   interactive attack + human review
├── evaluation.py final scores on the test split
├── config.py     all settings and word lists
└── requirements.txt
```

Before running, make sure `Model/` and `data/RomanUrduSentiment.csv` contain the real files.
Check with `ls -l Model data`: none of them should be 0 bytes.
`RomanUrduSentiment.csv` should have about 11000 lines (`wc -l data/RomanUrduSentiment.csv`).

## 3. Setup (once)

Linux/macOS:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check both of these:
```
python --version          # must say 3.12.x
pip show scikit-learn     # must say 1.8.0
```

If you see an `InconsistentVersionWarning` when running, your versions are wrong.
Delete `.venv` and repeat this step.

## 4. Review attacks (your main job)

```
python attacker.py
```

1. The first run creates `data/train.csv` and `data/test.csv` automatically.
   You should see `Created train.csv (8784) and test.csv (2196).`
2. Choose which attack to run this session from the menu.
3. For every generated sentence you see the original, the changed sentence, both predictions,
   and whether the attack succeeded.
4. Answer the two questions:
   - Does the changed sentence keep the original meaning? `1` = Yes, `2` = No
   - Where should it be stored?
     - **Training Dataset** (only offered if meaning is preserved)
     - **Attack Results Only**
     - **Reject Sample**
5. Press ENTER to go to the next one.

Type `q` at any prompt to stop. Progress is saved after every review.
Run the same command again to continue where you stopped.

Review rules:
- If the meaning changed, the sample can never go to training.
- Judge meaning only. Whether the attack succeeded should not influence your decision.
- Agree as a team on what counts as "same meaning" before you start.

## 5. Review attacks on the test split (for the meaning-preservation score)

```
python attacker.py test
```

This works the same way, but it only asks the meaning question. Test samples are never used for training.

## 6. Final scores

```
python evaluation.py
```

This takes about 15 seconds. It prints a table and saves `data/evaluation_summary.csv` with:
clean accuracy, robust accuracy, attack success rate, average queries per successful attack,
and the human meaning-preservation rate (from the reviews done in step 5).

## 7. Output files (in `data/`)

| File | Contents |
|---|---|
| `attack_results.csv` | Every reviewed attack with all metadata and your decisions |
| `adversarial_train.csv` | Only the samples you approved for training |
| `train.csv`, `test.csv` | Goku's exact 80/20 split, created automatically |
| `evaluation_summary.csv` | Final scores |

Never edit these files by hand while the attacker is running.
Labels are `1` = Positive and `0` = Negative.

## 8. Changing settings

Edit `config.py`:
- `ENABLED_ATTACKS`: switch an attack on or off (it appears in the session menu)
- `MAX_MODS`: maximum number of changes per sentence (default 2)
- `MAX_CANDIDATES`: maximum model queries per sentence per attack (default 30)
- `SEED`: random seed
- `SPELLING_VARIANTS`, `CODE_SWITCH_WORDS`, `HOMOGLYPHS`: the word lists

The spelling and code-switching lists are starter drafts. Verify every pair before relying on results,
and agree as a team before changing them, because everyone should review with the same lists.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `EmptyDataError: No columns to parse` | A file in `data/` is 0 bytes. Delete `train.csv`, `test.csv`, `attack_results.csv` and `adversarial_train.csv`, make sure `RomanUrduSentiment.csv` is not empty, and run again. |
| `InconsistentVersionWarning` | Wrong Python or scikit-learn version. Redo step 3 with Python 3.12. |
| `model file not found` | `svm.pkl` or `tfidf.pkl` is missing from `Model/`. |
| Strange characters in the terminal | Use a terminal with UTF-8 support (Windows Terminal, or any Linux/macOS terminal). |
