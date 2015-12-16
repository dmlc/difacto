#ifndef DIFACTO_TEST_CPP_UTILS_H_
#define DIFACTO_TEST_CPP_UTILS_H_
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

/**
 * \brief return a debug string of a vector
 */
template <typename V>
inline std::string DebugStr(const V* data, int n, int m = 5) {
  std::stringstream ss;
  ss << "[" << n << "]: ";
  if (n <= 2 * m) {
    for (int i = 0; i < n; ++i) ss << data[i] << " ";
  } else {
    for (int i = 0; i < m; ++i) ss << data[i] << " ";
    ss << "... ";
    for (int i = n-m; i < n; ++i) ss << data[i] << " ";
  }
  return ss.str();
}

}  // namespace difacto

#endif  // DIFACTO_TEST_CPP_UTILS_H_
