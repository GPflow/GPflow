.PHONY: help clean dev-install install package format test

help:
	@echo "The following make targets are available:"
	@echo "	dev-install		install all dependencies for dev environment and sets a egg link to the project sources"
	@echo "	install			install all dependencies and the project in the current environment"
	@echo "	package			build pip package"
	@echo "	test			run all tests in parallel"
	@echo "	clean			removes package, build files and egg info"

clean:
	rm -rf dist *.egg-info build

dev-install:
	pip install -e .

install:
	pip install .

package:
	python setup.py bdist

format:
	black -t py36 -l 100 gpflow tests doc setup.py

test:
	pytest -v --durations=10 tests/
