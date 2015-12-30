#include <gtest/gtest.h>
#include <map>
#include "./utils.h"
#include "common/kv_match.h"

// a referance impl based std::map
template <typename K, typename V>
size_t KVMatchRefer (
    const std::vector<K>& keys_a,
    const std::vector<V>& vals_a,
    const std::vector<K>& keys_b,
    const std::vector<V>& vals_b,
    std::vector<K>* joined_keys,
    std::vector<V>* joined_vals,
    int val_len = 1) {
  std::map<K, std::vector<V>> data;
  for (size_t i = 0; i < keys_a.size(); ++i) {
    auto& v = data[keys_a[i]];
    v.resize(val_len);
    for (int j = 0; j < val_len; ++j) {
      v[i] = vals_a[i*val_len +j];
    }
  }

  for (size_t i = 0; i < keys_b.size(); ++i) {
    auto it = data.find(keys_b[i]);
    if (it == data.end()) {
      auto& v = data[keys_b[i]];
      v.resize(val_len);
      for (int j = 0; j < val_len; ++j) {
        v[i] = vals_b[i*val_len +j];
      }
    } else {
      auto& v = it->second;
      for (int j = 0; j < val_len; ++j) {
        v[i] = vals_b[i*val_len +j];
      }
    }
  }

  for (auto it : data) {
    joined_keys->push_back(it->first);
    joined_vals->push_back(it->second.begin(), it->second.end());
  }
  return joined_keys->size();
}

std::uniform_int_distribution<int> dist_val(0, 1000);

void test(int n, int k) {
  std::vector<uint32_t> key1, key2, jkey1, jkey2;
  std::vector<int> val1, val2, jval1, jval2;
  gen_keys(n, &key1);
  gen_keys(n, &key2);
  for (size_t i = 0; i < key1.size(); ++i) {
    for (int j = 0; j < k; ++j)
      val1.push_back(dist_val(generator));
  }
  for (size_t i = 0; i < key2.size(); ++i) {
    for (int j = 0; j < k; ++j)
      val2.push_back(dist_val(generator));
  }


  size_t ret1 = KVUnion(key1, val1, key2, val2, &jkey1, &jkey2, k, PLUS, 4);
  size_t ret2 = KVUnionRefer(key1, val1, key2, val2, &jkey1, &jkey2, k);

  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(jval1.size(), jval2.size());
  EXPECT_EQ(jkey1.size(), jkey2.size());

  EXPECT_EQ(norm2(jkey1.data(), jkey1.size()),
            norm2(jkey2.data(), jkey2.size()));
  EXPECT_EQ(norm1(jval1.data(), jval1.size()),
            norm1(jval2.data(), jval2.size()));
}


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
