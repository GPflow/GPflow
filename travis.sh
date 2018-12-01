#!/bin/bash

set -ex


TESTRUN="pytest -W ignore::UserWarning --durations=10 -n auto --cov=./gpflow"


if [[ ${TEST_SUITE:-all} = all ]]; then
    ${TESTRUN} ./tests
elif [[ ${TEST_SUITE} = units ]]; then
    ${TESTRUN} -m 'not notebooks' ./tests
else
    ${TESTRUN} -m ${TEST_SUITE} ./tests
fi

codecov --token=2ae2a756-f39c-467c-bd9c-4bdb3dc439c8
