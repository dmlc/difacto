if [ ${TASK} == "lint" ]; then
    pip install cpplint --user `whoami`
fi
