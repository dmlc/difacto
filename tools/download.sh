#!/bin/bash
if [ $# -ne 1 ]; then
    echo "usage: $0 dataset_name"
    echo "  dataset_name can be rcv1, criteo, ctra, ..."
    echo "sample: $0 ctra"
    exit 0
fi

mkdir -p data && cd data

# download from http://data.dmlc.ml/difacto/datasets/
dmlc_download() {
    url=http://data.dmlc.ml/difacto/datasets/
    file=$1
    dir=`dirname $file`
    if [ ! -e $file ]; then
        wget ${url}/${file} -P ${dir}
    fi
}

# download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/
libsvm_download() {
    url=https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/
    file=$1
    if [ ! -e $file ]; then
        if [ ! -e ${file}.bz2 ]; then
            wget ${url}/${file}.bz2
        fi
        bzip2 -d ${file}.bz2
    fi
}
if [ $1 == "ctra" ]; then
    dmlc_download ctra/ctra_train.rec
    dmlc_download ctra/ctra_val.rec
elif [ $1 == "criteo" ]; then
    dmlc_download criteo_kaggle/criteo_train.rec
    dmlc_download criteo_kaggle/criteo_val.rec
elif [ $1 == "gisette" ]; then
    libsvm_download gisette_scale
    libsvm_download gisette_scale.t
elif [ $1 == "rcv1" ]; then
    libsvm_download rcv1_train.binary
else
    echo "unknown dataset name : $1"
fi
