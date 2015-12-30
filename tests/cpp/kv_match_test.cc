#include <gtest/gtest.h>
#include <unordered_map>
#include "./utils.h"
#include "common/kv_match.h"

using namespace difacto;

// a referance impl based std::map
template <typename K, typename V>
size_t KVMatchRefer (
  const std::vector<K>& src_key,
  const std::vector<V>& src_val,
  const std::vector<K>& dst_key,
  std::vector<V>* dst_val,
  int val_len = 1) {
  std::unordered_map<K, std::vector<V>> data;
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
      dst_val->at(i*val_len+j) = it->second[j];
    }
  }
  return matched;
}

std::uniform_int_distribution<int> dist_val(0, 1000);

void test(int n, int k) {
  std::vector<uint32_t> key1, key2;
  std::vector<int> val1, val2, val3;

  gen_keys(n, n*10, &key1);
  gen_keys(n, n*10, &key2);
  for (size_t i = 0; i < key1.size(); ++i) {
    for (int j = 0; j < k; ++j)
      val1.push_back(dist_val(generator));
  }

  size_t ret2 = KVMatchRefer(key1, val1, key2, &val3, k);
  size_t ret1 = KVMatch(key1, val1, key2, &val2, k, ASSIGN, 4);

  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(val2.size(), val3.size());
  EXPECT_EQ(norm1(val2.data(), val2.size()),
            norm1(val3.data(), val3.size()));
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
