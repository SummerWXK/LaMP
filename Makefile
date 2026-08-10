.PHONY: help check autoformat test
.DEFAULT_GOAL := help

help:
	@echo "make check       Run Black and Ruff without changing files"
	@echo "make autoformat  Format Python files and apply safe Ruff fixes"
	@echo "make test        Run CPU unit tests"

check:
	black --check starVLA scripts tests
	ruff check starVLA scripts tests

autoformat:
	black starVLA scripts tests
	ruff check --fix starVLA scripts tests

test:
	pytest -q
