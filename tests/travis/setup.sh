if [ ${TASK} == "lint" ]; then
    pip install cpplint pylint --user `whoami`
fi

# setup cache prefix
export CACHE_PREFIX=${HOME}/.cache/usr
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:${CACHE_PREFIX}/include
export C_INCLUDE_PATH=${C_INCLUDE_PATH}:${CACHE_PREFIX}/include
export LIBRARY_PATH=${LIBRARY_PATH}:${CACHE_PREFIX}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CACHE_PREFIX}/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${CACHE_PREFIX}/lib

if [ ${TASK} == "cpp-test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
fi
