.PHONY: setup lock data-pull mlflow-ui

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.lock

lock:
	.venv/bin/pip freeze > requirements.lock

data-pull:
	.venv/bin/dvc pull

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri mlruns
