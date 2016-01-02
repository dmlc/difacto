#include <gtest/gtest.h>
#include "data/data_store.h"
#include "./utils.h"

TEST(DataStore, Base) {

  std::vector<float>* a = new std::vector<float>();
  void* b = reinterpret_cast<void*>(a);
  std::vector<float>* c = reinterpret_cast<std::vector<float>*>(b);

  std::shared_ptr<std::vector<float>> f;
  std::shared_ptr<char> d;
  d = std::shared_ptr<char>(f, (char*)f->data());
  // d = std::make_shared<void>(new std::vector<float>());

}
