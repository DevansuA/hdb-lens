.PHONY: install data train test lint app docker

install:
	pip install -e ".[app,dev]"

data:
	python -m hdblens.ingest

train:
	python scripts/run_pipeline.py

test:
	pytest -q

lint:
	ruff check src tests app scripts

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t hdb-lens . && docker run -p 8501:8501 hdb-lens
