# Coffee Imbalance — RACOG vs SMOTE for Coffee Roast Quality

## 1) Setup
```bash
pip install -r requirements.txt
```

## 2) Phase 1 — Features & GAF
Put Artisan CSVs into `data/raw/`.

```bash
python scripts/make_features.py
```
Outputs:
- `data/processed/features_table.csv`
- `data/processed/*_gaf3ch_128.npy`

## 3) Labeling
Prepare `data/processed/labels_map.csv` with columns: `file_stem,label`.
Then run:
```bash
python scripts/label_features_by_csv.py
```

## 4) Phase 2 — Tabular CV
```bash
python scripts/run_tabular_cv.py
```

## 5) Grid Search
Open `scripts/run_grid_search.py` and set `mode` to `"smote"` or `"racog"`.
```bash
python scripts/run_grid_search.py
```

## 6) CNN(GAF) CV (optional)
Integrate `src/models/cnn_resnet.py` in your notebook or add a script.

## 7) Train Final & External Evaluation (No Leakage)
After picking best settings from CV/grid:
```bash
python scripts/train_final.py
python scripts/eval_external.py
```
