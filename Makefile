.PHONY: all test test-frontend test-backend run lint clean

all: test

test: test-frontend test-backend

test-frontend:
	node --test tests/test_document_view.cjs

test-backend:
	pytest

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"

