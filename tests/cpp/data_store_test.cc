#include <gtest/gtest.h>
#include "data/data_store.h"
#include "data/batch_iter.h"
#include "./utils.h"

using namespace difacto;

TEST(DataStore, MemBase) {
  DataStore store;
  int n = 1000;
  std::vector<real_t> val1;
  std::vector<int> val2;
  std::vector<uint64_t> val3;

  gen_vals(n, -100, 100, &val1);
  gen_vals(n, -100, 100, &val2);
  gen_vals(n, -100, 100, &val3);

  store.Store("1", val1.data(), val1.size());
  store.Store("2", val2.data(), val2.size());

  SArray<real_t> ret1;
  SArray<int> ret2;
  store.Fetch("1", &ret1);
  store.Fetch("2", &ret2, Range(10,30));

  // overwrite key
  SArray<uint64_t> ret3;
  store.Store("1", val3.data(), val3.size());
  store.Fetch("1", &ret3);

  // noncopy
  {
    SArray<int> val4(val2);
    store.Store("4", val4);
  }
  SArray<int> ret4;
  store.Fetch("4", &ret4);

  EXPECT_EQ(norm2(val1), norm2(ret1));
  EXPECT_EQ(norm2(SArray<int>(val2).segment(10, 30)), norm2(ret2));
  EXPECT_EQ(norm2(val3), norm2(ret3));
  EXPECT_EQ(norm2(val2), norm2(ret4));
}

TEST(DataStore, RowBlock) {
  BatchIter iter("../tests/data", "libsvm", 0, 1, 100);
  CHECK(iter.Next());
  auto data = iter.Value();

  DataStore store;
  store.Store("1", data);

  SharedRowBlockContainer<feaid_t> blk1;
  store.Fetch("1", &blk1);
  check_equal(data, blk1.GetBlock());

  SharedRowBlockContainer<feaid_t> blk2;
  store.Fetch("1", &blk2, Range(10, 40));
  check_equal(data.Slice(10, 40), blk2.GetBlock());
}
