# Makefile for gbtm_project
# Run `make help` for a list of available targets.

.PHONY: help test lint coverage simulate benchmark clean

help:
	@echo "Available targets:"
	@echo "  make test       Run the fast test suite (skips slow/recovery tests)"
	@echo "  make lint       Run flake8 linter"
	@echo "  make coverage   Run fast tests with HTML coverage report"
	@echo "  make simulate   Run the simulation scripts standalone"
	@echo "  make benchmark  Run only the Cambridge benchmark tests"
	@echo "  make clean      Remove __pycache__, coverage artifacts, htmlcov"

test:
	pytest -v --tb=short -m "not slow"

lint:
	flake8 main.py app.py tests/

coverage:
	pytest -v --tb=short -m "not slow" \
		--cov=main \
		--cov-report=term-missing \
		--cov-report=html:htmlcov
	@echo "HTML report written to htmlcov/index.html"

simulate:
	python tests/simulate.py

benchmark:
	pytest -v --tb=short tests/test_cambridge_benchmark.py -m "not slow"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -rf htmlcov .coverage coverage.xml
