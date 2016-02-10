/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include <map>
#include "./utils.h"
#include "common/kv_union.h"

using namespace difacto;

// a referance impl based std::map
template <typename K, typename V>
void KVUnionRefer(
    const SArray<K>& keys_a,
    const SArray<V>& vals_a,
    const SArray<K>& keys_b,
    const SArray<V>& vals_b,
    SArray<K>* joined_keys,
    SArray<V>* joined_vals,
    int val_len = 1) {
  std::map<K, SArray<V>> data;
  for (size_t i = 0; i < keys_a.size(); ++i) {
    auto& v = data[keys_a[i]];
    v.resize(val_len);
    for (int j = 0; j < val_len; ++j) {
      v[j] = vals_a[i*val_len +j];
    }
  }

  for (size_t i = 0; i < keys_b.size(); ++i) {
    auto it = data.find(keys_b[i]);
    if (it == data.end()) {
      auto& v = data[keys_b[i]];
      v.resize(val_len);
      for (int j = 0; j < val_len; ++j) {
        v[j] = vals_b[i*val_len +j];
      }
    } else {
      auto& v = it->second;
      for (int j = 0; j < val_len; ++j) {
        v[j] += vals_b[i*val_len +j];
      }
    }
  }

  for (auto it : data) {
    joined_keys->push_back(it.first);
    for (V v : it.second) joined_vals->push_back(v);
  }
}

namespace  {
void test(int n, int k) {
  SArray<uint32_t> key1, key2, jkey1, jkey2;
  SArray<real_t> val1, val2, jval1, jval2;
  gen_keys(n, n*10, &key1);
  gen_keys(n, n*10, &key2);
  gen_vals(key1.size()*k, -100, 100, &val1);
  gen_vals(key2.size()*k, -100, 100, &val2);

  KVUnion(key1, val1, key2, val2, &jkey1, &jval1, PLUS, 4);
  KVUnionRefer(key1, val1, key2, val2, &jkey2, &jval2, k);

  EXPECT_EQ(jval1.size(), jval2.size());
  EXPECT_EQ(jkey1.size(), jkey2.size());

  EXPECT_EQ(norm2(jkey1.data(), jkey1.size()),
            norm2(jkey2.data(), jkey2.size()));
  EXPECT_EQ(norm1(jval1.data(), jval1.size()),
            norm1(jval2.data(), jval2.size()));
}
}  // namespace


TEST(KVUnion, Union) {
  for (int i = 0; i < 10; ++i) {
    test(1000, 1);
  }
}

TEST(KVUnion, Val3) {
  for (int i = 0; i < 10; ++i) {
    test(1000, 4);
  }
}
