#!/bin/bash

if [ ${TASK} == "lint" ]; then
    make lint
    exit $?
fi

if [ ${TASK} == "cpp-test" ]; then
    make -j4 cpp-test CXX=g++-4.8
    cd build; find *_test -exec ./{} \;
fi
