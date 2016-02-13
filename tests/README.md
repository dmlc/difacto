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

- compile by `make test` on the project root directory

- run all tests by
  ```bash
  cd build; ./difacto_tests
  ```

  Use `./difacto_tests --gtest_list_tests` to list all tests and
  `./difacto_tests --gtest_filter=PATTERN` to run some particular tests

## coverage

  ```bash
  make clean; make -j8 test ADD_CFLAGS=-converage; cd build; ./difacto_tests; codecov
  ```

```bash
make clean; make -j8 NO_REVERSE_ID=1
```
## matlab tests

  Some scripts used to generate the *ground truth*.
