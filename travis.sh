#!/bin/bash

set -ex

# TESTRUN="pytest -W ignore::UserWarning --durations=5 -n 4 --cov=./gpflow"

TESTRUN="pytest -W ignore::UserWarning --durations=10 -d --tx 3*popen//python=python3.6 --cov=./gpflow"



if [[ ! ${TRAVIS_BRANCH} =~ ^(master|develop)$ ]]; then
    ${TESTRUN} '--skipslow' ./tests
# Special case for PRs from develop to master
elif [[ ${TRAVIS_PULL_REQUEST} != false ]] && [[ ${TRAVIS_PULL_REQUEST_BRANCH} != master ]]; then
    ${TESTRUN} '--skipslow' ./tests
else
    ${TESTRUN} ./tests
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8
