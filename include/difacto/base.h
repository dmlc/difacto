/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_BASE_H_
#define DIFACTO_BASE_H_
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
#include "dmlc/logging.h"
namespace difacto {
/*!
 * \brief use float as the weight and gradient type
 */
typedef float real_t;
/*!
 * \brief use uint64_t as the feature index type
 */
typedef uint64_t feaid_t;
/**
 * \brief a list of keyword arguments
 */
typedef std::vector<std::pair<std::string, std::string>> KWArgs;
/**
 * \brief the default number of threads
 */
#define DEFAULT_NTHREADS 2
/**
 * \brief reverse the bytes of x to make it more uniformly spanning the space
 * \param x the feature index
 */
inline feaid_t ReverseBytes(feaid_t x) {
  // return x;
  x = x << 32 | x >> 32;
  x = (x & 0x0000FFFF0000FFFFULL) << 16 |
      (x & 0xFFFF0000FFFF0000ULL) >> 16;
  x = (x & 0x00FF00FF00FF00FFULL) << 8 |
      (x & 0xFF00FF00FF00FF00ULL) >> 8;
  x = (x & 0x0F0F0F0F0F0F0F0FULL) << 4 |
      (x & 0xF0F0F0F0F0F0F0F0ULL) >> 4;
  return x;
}
/**
 * \brief generate a new feature index containing the feature group id
 *
 * @param x the feature index
 * @param gid feature group id
 * @param nbits number of bits used to encode gid
 * @return the new feature index
 */
inline feaid_t EncodeFeaGrpID(feaid_t x, int gid, int nbits = 12) {
  CHECK_GE(gid, 0); CHECK_LT(gid, 1 << nbits);
  return (x << nbits) | gid;
}
/**
 * \brief get the feature group id from a feature index
 *
 * @param x the feature index
 * @param nbits number of bits used to encode gid
 * @return the feature group id
 */
inline feaid_t DecodeFeaGrpID(feaid_t x, int nbits = 12) {
  return x % (1 << nbits);
}
/**
 * \brief returns true if it is currently under distributed running
 */
inline bool IsDistributed() {
  return getenv("DMLC_ROLE") != nullptr;
}
/**
 * \brief for debug printing
 */
#define LL LOG(ERROR)
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
/**
 * \brief return a debug string of a vector
 */
template <typename Vec>
inline std::string DebugStr(const Vec& vec) {
  return DebugStr(vec.data(), vec.size());
}

}  // namespace difacto
#endif  // DIFACTO_BASE_H_
