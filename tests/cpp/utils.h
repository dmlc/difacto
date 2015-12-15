#ifndef DIFACTO_TEST_CPP_UTILS_H_
#define DIFACTO_TEST_CPP_UTILS_H_
#include <math.h>
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

}  // namespace difacto

#endif  // DIFACTO_TEST_CPP_UTILS_H_
