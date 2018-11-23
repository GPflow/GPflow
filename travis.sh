#!/bin/bash

set -e

PYTEST_RUN="pytest -W ignore::UserWarning --durations=5 --cov=./gpflow -n auto"

if [ "$TRAVIS_BRANCH" = "develop" ] || [ "$TRAVIS_BRANCH" = "master" ]; then
    $($PYTEST_RUN ./tests)
else
    $($PYTEST_RUN -k 'not notebooks' ./tests)
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8