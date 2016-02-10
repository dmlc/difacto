#!/bin/bash

if [ ${TASK} == "lint" ]; then
    make lint
    exit $?
fi

if [ ${TASK} == "cpp-test" ]; then
    make -j4 test CXX=g++-4.8 -coverage
    cd build; find *_test -exec ./{} \;
fi
