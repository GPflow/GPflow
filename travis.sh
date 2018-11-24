#!/bin/bash

set -ev

TESTRUN="pytest -W ignore::UserWarning --durations=5 --cov=./gpflow -n auto"

echo ${TRAVIS_BRANCH}
echo ${TRAVIS_PULL_REQUEST}
echo ${TRAVIS_PULL_REQUEST_BRANCH}

if [ "${TRAVIS_PULL_REQUEST}" != "false" ] && [ "${TRAVIS_PULL_REQUEST_BRANCH}" != "develop" ]; then
    ${TESTRUN} -k 'not notebooks' ./tests;
else
    ${TESTRUN} ./tests;
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8
