# test codes

## test data

[data](data) contains the first 100 lines of the
[rcv1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary)
dataset in libsvm format

## c++ tests
[cpp/](cpp/) contains c++ test codes

- `gtest` is needed for compiling. One can install `libgtest-dev` using a package
  manager, e.g. for ubuntu

  ```bash
  sudo apt-get install libgtest-dev
  cd /usr/src/gtest
  sudo cmake .
  sudo make
  sudo mv libg* /usr/lib/
  ```

- compile by `make cpp-test` on the project root directory

- test by
  ```bash
  cd ../build
  find *_test -exec ./{} \;
  ```


## travis
