if [ ${TASK} == "lint" ]; then
    pip install cpplint pylint --user `whoami`
fi
