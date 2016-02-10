/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef TESTS_CPP_UTILS_H_
#define TESTS_CPP_UTILS_H_
#include <math.h>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <sstream>
#include "dmlc/data.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
#include "data/localizer.h"
#include "reader/batch_reader.h"
namespace difacto {

template <typename T>
using RowBlock = dmlc::RowBlock<T>;

/**
 * \brief returns the sum of a a vector
 */
template <typename T>
T sum(T const* data, int len) {
  T res = 0;
  for (int i = 0; i < len; ++i) res += data[i];
  return res;
}

/**
 * \brief return l1 norm of a vector
 */
template <typename T>
T norm1(T const* data, int len) {
  T norm = 0;
  for (int i = 0; i < len; ++i) norm += fabs(data[i]);
  return norm;
}

/**
 * \brief return l2 norm of a vector
 */
template <typename T>
double norm2(T const* data, int len) {
  double norm = 0;
  for (int i = 0; i < len; ++i) norm += data[i] * data[i];
  return norm;
}

template <typename T>
double norm2(const T& data) {
  return norm2(data.data(), data.size());
}


std::default_random_engine generator;

/**
 * \brief generate a list of unique and sorted keys
 * \param key_len the expected key length
 * \param max_key the maximal key size
 */
void gen_keys(int key_len, uint32_t max_key, SArray<uint32_t>* key) {
  key->resize(key_len);
  std::uniform_int_distribution<uint32_t> distribution(
      0, max_key);
  for (int i = 0; i < key_len; ++i) {
    (*key)[i] = distribution(generator);
  }
  std::sort(key->begin(), key->end());
  auto end = std::unique(key->begin(), key->end());
  key->resize(std::distance(key->begin(), end));
}

/**
 * \brief generate a list of random values
 *
 * @param len the length
 * @param max_val random value in [min_val, max_val)
 * @param val results
 */
template <typename V>
void gen_vals(int len, real_t min_val, real_t max_val, SArray<V>* val) {
  val->resize(len);
  std::uniform_real_distribution<real_t> dis(min_val, max_val);
  for (int i = 0; i < len; ++i) {
    (*val)[i] = static_cast<V>(dis(generator));
  }
}

/**
 * \brief generate a list of random values
 *
 * @param len the length
 * @param max_val random value in [min_val, max_val)
 * @param val results
 */
template <typename V>
void gen_vals(int len, SArray<V>* val, real_t min_val = -10, real_t max_val = 10) {
  val->resize(len);
  std::uniform_real_distribution<real_t> dis(min_val, max_val);
  for (int i = 0; i < len; ++i) {
    (*val)[i] = static_cast<V>(dis(generator));
  }
}

/**
 * \brief check rowblock a == b
 */
template <typename T>
void check_equal(RowBlock<T> a, RowBlock<T> b) {
  EXPECT_EQ(a.size, b.size);
  EXPECT_EQ(a.label != nullptr, b.label != nullptr);
  EXPECT_EQ(norm1(a.offset, a.size+1) - a.offset[0]*(a.size+1),
            norm1(b.offset, b.size+1) - b.offset[0]*(b.size+1));
  if (a.label) {
    EXPECT_EQ(norm1(a.label, a.size), norm1(b.label, b.size));
  }
  size_t nnz = a.offset[a.size] - a.offset[0];
  EXPECT_EQ(nnz, b.offset[b.size] - b.offset[0]);
  EXPECT_EQ(norm1(a.index+a.offset[0], nnz), norm1(b.index+b.offset[0], nnz));
  EXPECT_EQ(a.value != nullptr, b.value != nullptr);
  if (a.value) {
    EXPECT_EQ(norm2(a.value+a.offset[0], nnz), norm2(b.value+b.offset[0], nnz));
  }
}

/**
 * \brief load ../tests/data
 */
void load_data(dmlc::data::RowBlockContainer<unsigned>* data,
               std::vector<feaid_t>* uidx) {
  CHECK_NOTNULL(data);
  BatchReader reader("../tests/data", "libsvm", 0, 1, 100);
  CHECK(reader.Next());
  Localizer lc;
  lc.Compact(reader.Value(), data, uidx);
  if (uidx) {
    for (auto& idx : *uidx) idx = ReverseBytes(idx);
  }
}

}  // namespace difacto

#endif  // TESTS_CPP_UTILS_H_
