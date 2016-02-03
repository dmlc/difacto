#!/bin/bash
if [ $# -ne 2 ]; then
    echo "usage: $0 dataset_name local_dir"
    echo "  dataset_name can be rcv1, criteo, ctra, ..."
    echo "sample: $0 ctra data/"
    exit 0
fi

dir=$2
mkdir -p $dir
cd $dir

# download from http://data.dmlc.ml/difacto/datasets/
dmlc_download() {
    url=http://data.dmlc.ml/difacto/datasets/
    file=$1
    if [ ! -e $file ]; then
        if [ ! -e ${file}.gz ]; then
            wget ${url}/${file}.gz
            wget ${url}/${file}.gz.md5
            md5sum -c ${file}.gz.md5
            rm ${file}.gz.md5
        fi
        gunzip ${file}.gz
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
    dmlc_download ctra_train
    dmlc_download ctra_test
elif [ $1 == "gisette" ]; then
    libsvm_download gisette_scale
    libsvm_download gisette_scale.t
elif [ $1 == "rcv1" ]; then
    libsvm_download rcv1_train.binary
else
    echo "unknown dataset name : $1"
fi
