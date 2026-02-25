ENV_NAME=solana-research
PY=conda run -n $(ENV_NAME) python

.PHONY: setup fetch build-features train backtest dashboard

setup:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

fetch:
	$(PY) -m src.data.fetch_ohlcv --config configs/default.yaml

build-features:
	$(PY) -m src.features.build_features --config configs/default.yaml

train:
	$(PY) -m src.models.train_models --config configs/default.yaml

backtest:
	$(PY) -m src.backtest.walkforward --config configs/default.yaml

dashboard:
	conda run -n $(ENV_NAME) streamlit run src/dashboard/app.py -- --config configs/default.yaml
