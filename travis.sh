#!/bin/bash

set -ex


TESTRUN="pytest -W ignore::UserWarning --durations=5 --cov=./gpflow -n auto"

echo "TRAVIS_COMMIT_MESSAGE=${TRAVIS_COMMIT_MESSAGE}"
echo "TRAVIS_BRANCH=${TRAVIS_BRANCH}"
echo "TRAVIS_PULL_REQUEST=${TRAVIS_PULL_REQUEST}"
echo "TRAVIS_PULL_REQUEST_BRANCH=${TRAVIS_PULL_REQUEST_BRANCH}"
echo $(git status -s -b)

if [[ ! ${TRAVIS_BRANCH} =~ ^(master|develop)$ ]]; then
    ${TESTRUN} ./tests
# Special case for PRs from develop to master
elif [[ ${TRAVIS_PULL_REQUEST} != "false" ]] && [[ ${TRAVIS_PULL_REQUEST_BRANCH} != "master" ]]; then
    ${TESTRUN} ./tests
else
    ${TESTRUN} -k 'not notebooks' ./tests
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8
