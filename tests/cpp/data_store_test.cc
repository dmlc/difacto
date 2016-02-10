/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "dmlc/memory_io.h"
#include "data/data_store.h"
#include "./utils.h"

using namespace difacto;

TEST(DataStore, MemBase) {
  DataStore store;
  int n = 1000;
  SArray<real_t> val1;
  SArray<int> val2;
  SArray<uint64_t> val3;

  gen_vals(n, -100, 100, &val1);
  gen_vals(n, -100, 100, &val2);
  gen_vals(n, -100, 100, &val3);

  store.Store("1", val1.data(), val1.size());
  store.Store("2", val2.data(), val2.size());

  SArray<real_t> ret1;
  SArray<int> ret2;
  store.Fetch("1", &ret1);
  store.Fetch("2", &ret2, Range(10, 30));

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

TEST(DataStore, Meta) {
  DataStore store;
  int n = 1000;
  SArray<real_t> val1;
  SArray<int> val2;
  SArray<uint64_t> val3;
  gen_vals(n, -100, 100, &val1);
  gen_vals(n, -100, 100, &val2);
  gen_vals(n, -100, 100, &val3);


  store.Store("1", val1);
  store.Store("2", val2);
  store.Store("3", val3);

  std::string meta;
  dmlc::Stream* os = new dmlc::MemoryStringStream(&meta);
  store.Save(os);
  delete os;

  LL << meta;

  DataStore store2;
  dmlc::Stream* is = new dmlc::MemoryStringStream(&meta);
  store2.Load(is);
  delete is;

  EXPECT_EQ(store2.size("1"), n);
  EXPECT_EQ(store2.size("2"), n);
  EXPECT_EQ(store2.size("3"), n);
}
