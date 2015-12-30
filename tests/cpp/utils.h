#ifndef DIFACTO_TEST_CPP_UTILS_H_
#define DIFACTO_TEST_CPP_UTILS_H_
#include <random>
#include <limits>
#include <algorithm>
#include <math.h>
#include <sstream>
namespace difacto {

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

std::default_random_engine generator;

/**
 * \brief generate a list of unique and sorted keys
 * \param key_len the expected key length
 * \param max_key the maximal key size
 */
void gen_keys(int key_len, uint32_t max_key, std::vector<uint32_t>* key) {
  key->resize(key_len);
  std::uniform_int_distribution<uint32_t> distribution(
      0, max_key);
  for (int i = 0; i < key_len; ++i) {
    key->at(i) = distribution(generator);
  }
  std::sort(key->begin(), key->end());
  auto end = std::unique(key->begin(), key->end());
  key->resize(std::distance(key->begin(), end));
}

}  // namespace difacto

#endif  // DIFACTO_TEST_CPP_UTILS_H_
