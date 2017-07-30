#!/bin/bash

# Script for running GPflow tests in sequential and parallel modes.
# Running tensorflow based tests in distinct processes prevents
# bad memory accumulations which can lead to crashes or slow runs
# on resource limited hardware.
# Written by Artem Artemev, 06/08/2017

set -e

mode=${1:-"--sequential"}

case "$mode" in
    -p|--parallel)
    numproc=$([[ $(uname) == 'Darwin' ]] && sysctl -n hw.physicalcpu_max || nproc)
    echo ">>> Parallel mode. Number of processes = $numproc"
    echo testing/test_*.py | xargs -n 1 -P "$numproc" bash -c 'nosetests -v --nologcapture $0 || exit 255'
    ;;
    -s|--sequential)
    for test_file in testing/test_*.py; do
      echo ">>> Run $test_file"
      nosetests -v --nologcapture "$test_file"
      rc=$?
      if [ "$rc" != "0" ]; then
        echo ">>> $test_file failed"
        exit $rc
      fi
    done
    ;;
    *)
    ;;
esac
