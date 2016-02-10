/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include <unordered_map>
#include "./utils.h"
#include "common/kv_match.h"
using namespace difacto;

// a referance impl based std::map
template <typename K, typename V>
size_t KVMatchRefer(
    const SArray<K>& src_key,
    const SArray<V>& src_val,
    const SArray<K>& dst_key,
    SArray<V>* dst_val,
    int val_len = 1) {
  std::unordered_map<K, SArray<V>> data;
  for (size_t i = 0; i < src_key.size(); ++i) {
    auto& v = data[src_key[i]];
    v.resize(val_len);
    for (int j = 0; j < val_len; ++j) {
      v[j] = src_val[i*val_len +j];
    }
  }
  dst_val->resize(dst_key.size() * val_len);
  size_t matched = 0;
  for (size_t i = 0; i < dst_key.size(); ++i) {
    auto it = data.find(dst_key[i]);
    if (it == data.end()) continue;
    matched += val_len;
    for (int j = 0; j < val_len; ++j) {
      (*dst_val)[i*val_len+j] = it->second[j];
    }
  }
  return matched;
}

void test(int n, int k) {
  SArray<uint32_t> key1, key2;
  SArray<real_t> val1, val2, val3;

  gen_keys(n, n*10, &key1);
  gen_keys(n, n*10, &key2);
  gen_vals(key1.size()*k, -100, 100, &val1);

  size_t ret2 = KVMatchRefer(key1, val1, key2, &val3, k);
  size_t ret1 = KVMatch(key1, val1, key2, &val2, ASSIGN, 4);

  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(val2.size(), val3.size());
  EXPECT_EQ(norm2(val2), norm2(val3));
}


TEST(KVMatch, Match) {
  for (int i = 0; i < 10; ++i) {
    test(1000, 1);
  }
}

TEST(KVMatch, Val3) {
  for (int i = 0; i < 10; ++i) {
    test(1000, 4);
  }
}

// a referance impl based std::map
template <typename K, typename V>
size_t KVMatchVLenRefer(
    const SArray<K>& src_key,
    const SArray<V>& src_val,
    const SArray<int>& src_len,
    const SArray<K>& dst_key,
    SArray<V>* dst_val,
    SArray<int>* dst_len) {
  // insert
  std::unordered_map<K, SArray<V>> data;
  size_t n = 0;
  for (size_t i = 0; i < src_key.size(); ++i) {
    auto& v = data[src_key[i]];
    v.resize(src_len[i]);
    for (size_t j = 0; j < v.size(); ++j) {
      v[j] = src_val[n++];
    }
  }

  // match len
  n = 0;
  dst_len->resize(dst_key.size(), 0);
  for (size_t i = 0; i < dst_key.size(); ++i) {
    auto it = data.find(dst_key[i]);
    if (it == data.end()) continue;
    (*dst_len)[i] = it->second.size();
    n += it->second.size();
  }

  // match val
  dst_val->resize(n);
  n = 0;
  for (size_t i = 0; i < dst_key.size(); ++i) {
    auto it = data.find(dst_key[i]);
    if (it == data.end()) continue;
    for (int j = 0; j < it->second.size(); ++j) {
      (*dst_val)[n++] = it->second[j];
    }
  }
  return n;
}


void test_vlen(int n) {
  SArray<uint32_t> key1, key2;
  SArray<int> len1, len2, len3;
  SArray<real_t> val1, val2, val3;

  gen_keys(n, n*5, &key1);
  gen_keys(n, n*5, &key2);
  gen_vals(key1.size(), 0, 10, &len1);
  size_t m = 0;
  for (int i : len1) m += i;
  gen_vals(m, -10, 10, &val1);

  size_t ret2 = KVMatchVLenRefer(key1, val1, len1, key2, &val2, &len2);
  size_t ret1 = KVMatch(key1, val1, len1, key2, &val3, &len3);
  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(val2.size(), val3.size());
  EXPECT_EQ(norm2(val2), norm2(val3));
  EXPECT_EQ(len2.size(), len3.size());
  EXPECT_EQ(norm2(len2), norm2(len3));
}

TEST(KVMatch, VaryLength) {
  for (int i = 0; i < 10; ++i) {
    test_vlen(1000);
  }
}
