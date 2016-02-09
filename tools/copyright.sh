#!/bin/bash
# add copyright
dir=`pwd`/`dirname $0`/..; cd $dir
copyright=/tmp/copyright

cat >$copyright <<EOF
/**
 *  Copyright (c) 2015 by Contributors
 */
EOF
for file in `find include src tests -name "*.cc" -o -name "*.h"`; do
    if ! grep -q Copyright $file; then
        echo $file
        cat $copyright $file >$file.new && mv $file.new $file
    fi
done
