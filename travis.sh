#!/bin/bash

set -ex

TESTRUN="pytest -W ignore::UserWarning --durations=5 --cov=./gpflow -n auto"

if [[ ! ${TRAVIS_BRANCH} =~ ^(master|develop)$ ]]; then
    ${TESTRUN} '--skipslow' ./tests
# Special case for PRs from develop to master
elif [[ ${TRAVIS_PULL_REQUEST} != false ]] && [[ ${TRAVIS_PULL_REQUEST_BRANCH} != master ]]; then
    ${TESTRUN} '--skipslow' ./tests
else
    ${TESTRUN} ./tests
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8
