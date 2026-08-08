.PHONY: install build model warehouse app test clean

install:
	python -m pip install -r requirements.txt

warehouse:
	PYTHONPATH=src python scripts/build_warehouse.py

model:
	PYTHONPATH=src python scripts/train_models.py

build: warehouse model
	PYTHONPATH=src python scripts/build_dashboard_assets.py

app:
	streamlit run dashboard/app.py

test:
	PYTHONPATH=src pytest -q

clean:
	rm -f data/retention.duckdb artifacts/*.joblib
