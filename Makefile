.PHONY: setup audit test pipeline compare app clean

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

audit:
	uv run python scripts/audit.py

test:
	uv run pytest

pipeline:
	uv run python scripts/build_pipeline.py

compare:
	uv run python scripts/compare_models.py

app:
	uv run streamlit run src/proxypheno/app/streamlit_app.py

clean:
	rm -rf .venv __pycache__ .pytest_cache
