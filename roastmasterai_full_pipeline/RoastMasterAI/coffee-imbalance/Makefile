PY=python

setup:
	$(PY) -m pip install -e . --config-settings editable_mode=compat
	ruff check . || true

features:
	$(PY) scripts/make_features.py

validate:
	$(PY) scripts/validate_labels.py

cv:
	$(PY) scripts/run_tabular_cv.py

grid:
	$(PY) scripts/run_grid_search.py

final:
	$(PY) scripts/train_final.py

external:
	$(PY) scripts/eval_external.py

report:
	$(PY) scripts/make_report_figs.py
