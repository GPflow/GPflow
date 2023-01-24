NOTEBOOK_BLACK_CONFIG=-t py37 -l 80
NOTEBOOK_BLACK_TARGETS=doc/sphinx/notebooks
NOTEBOOK_ISORT_CONFIG=--atomic -l 80 --trailing-comma --remove-redundant-aliases --multi-line 3
NOTEBOOK_ISORT_TARGETS=doc/sphinx/notebooks

BLACK_CONFIG=-t py37 -l 100
BLACK_TARGETS=gpflow tests benchmark doc/*.py setup.py
ISORT_CONFIG=--atomic -l 100 --trailing-comma --remove-redundant-aliases --multi-line 3
ISORT_TARGETS=gpflow tests benchmark doc/*.py setup.py
MYPY_TARGETS=gpflow tests benchmark doc/*.py doc/sphinx/notebooks/getting_started setup.py

.PHONY: help clean dev-install install package format format-check type-check test check-all

help:
	@echo "The following make targets are available:"
	@echo "	dev-install		install all dependencies for dev environment and sets a egg link to the project sources"
	@echo "	install			install all dependencies and the project in the current environment"
	@echo "	package			build pip package"
	@echo "	clean			removes package, build files and egg info"
	@echo "	format			auto-format code"
	@echo "	format-check		check that code has been formatted correctly"
	@echo "	type-check		check that mypy is happy with type annotations"
	@echo "	test			run all tests"
	@echo "	check-all		run format-check, type-check, and test (as run by the continuous integration system)"

clean:
	rm -rf dist *.egg-info build

dev-install:
	pip install -r tests_requirements.txt -e .

install:
	pip install .

package:
	python setup.py bdist

format:
	black $(BLACK_CONFIG) $(BLACK_TARGETS)
	black $(NOTEBOOK_BLACK_CONFIG) $(NOTEBOOK_BLACK_TARGETS)
	isort $(ISORT_CONFIG) $(ISORT_TARGETS)
	isort $(NOTEBOOK_ISORT_CONFIG) $(NOTEBOOK_ISORT_TARGETS)

format-check:
	black --check $(BLACK_CONFIG) $(BLACK_TARGETS)
	black --check $(NOTEBOOK_BLACK_CONFIG) $(NOTEBOOK_BLACK_TARGETS)
	isort --check-only $(ISORT_CONFIG) $(ISORT_TARGETS)
	isort --check-only $(NOTEBOOK_ISORT_CONFIG) $(NOTEBOOK_ISORT_TARGETS)

type-check:
	mypy `python -m gpflow.mypy_flags` $(MYPY_TARGETS)

test:
	pytest -n auto --dist loadfile -v --durations=10 tests/

check-all: format-check type-check test
