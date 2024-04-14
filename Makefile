all: install

.PHONY: test

install:
	@echo "Installing Movement..."
	@python3 -m pip install -e .
	@echo "Done."
	@echo "Setting up pre-commit..."
	@pre-commit install
	@echo "Done."

lint:
	@echo "Checking lint..."
	@flake8
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@python3 -m pytest -v -n auto --durations=20 test
	@echo "PASS"

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --source=movement -m pytest -v test
	@python3 -m coverage html
	@echo "Done."

tree:
	@tree -d .
