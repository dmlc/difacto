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
std::uniform_int_distribution<uint32_t> distribution(
    0, std::numeric_limits<uint32_t>::max());

/**
 * \brief generate a list of unique and sorted keys
 */
void gen_keys(int n, std::vector<uint32_t>* key) {
  key->resize(n);
  for (int i = 0; i < n; ++i) {
    key->at(i) = distribution(generator);
  }
  std::sort(key->begin(), key->end());
  auto end = std::unique(key->begin(), key->end());
  key->resize(std::distance(key->begin(), end));
}

}  // namespace difacto

#endif  // DIFACTO_TEST_CPP_UTILS_H_
