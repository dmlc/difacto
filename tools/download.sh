#!/bin/bash
if [ $# -ne 2 ]; then
    echo "usage: $0 dataset_name local_dir"
    echo "  dataset_name includes rcv1, criteo, ctra, ..."
    exit 0
fi

dir=$2
mkdir -p $dir
cd $dir

if [ $1 == "ctra" ]; then
    if [ ! -e ctra_train ] || [ ! -e ctra_test ]; then
        if [ ! -e ctra.tar.gz ]; then
            wget https://cmu.box.com/shared/static/tolqotsal8d5whkiks8v8rk9lueqksq3.gz
            mv tolqotsal8d5whkiks8v8rk9lueqksq3.gz ctra.tar.gz
        fi
        tar -zxvf ctra.tar.gz
    fi
elif [ $1 == "ctrb" ]; then
    if [ ! -e ctrb_train ] || [ ! -e ctrb_test ]; then
        if [ ! -e ctrb.tar.gz ]; then
            wget https://cmu.box.com/shared/static/uvjuon4h9av3bz0fmai7n44cgar021xa.gz
            mv uvjuon4h9av3bz0fmai7n44cgar021xa.gz ctrb.tar.gz
        fi
        tar -zxvf ctrb.tar.gz
    fi
fi
