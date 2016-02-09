/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_FIND_POSITION_H_
#define DIFACTO_COMMON_FIND_POSITION_H_
#include <vector>
#include <thread>
#include "dmlc/logging.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
#include "./range.h"
namespace difacto {

namespace {
template <typename K>
size_t FindPosition(K const* src_begin, K const* src_end,
                    K const* dst_begin, K const* dst_end,
                    int* pos_begin, int* pos_end) {
  size_t n = 0;
  K const* src = std::lower_bound(src_begin, src_end, *dst_begin);
  K const* dst = std::lower_bound(dst_begin, dst_end, *src);

  int *pos = pos_begin + (dst - dst_begin);
  while (pos_begin != pos) { *pos_begin = -1; ++pos_begin; }
  while (src != src_end && dst != dst_end) {
    if (*src < *dst) {
      ++src;
    } else {
      if (!(*dst < *src)) {  // equal
        *pos = static_cast<int>(src - src_begin);
        ++src; ++n;
      } else {
        *pos = -1;
      }
      ++dst; ++pos;
    }
  }
  while (pos != pos_end) {*pos = -1; ++pos; }
  return n;
}

}  // namespace

/**
 * \brief store the position of dst[i] in src into pos[i], namely src[pos[i]] == dst[i]
 *
 * @param src unique and sorted vector
 * @param dst unique and sorted vector
 * @param pos the positions, -1 means no matched
 * @return the number of matched
 */
template <typename K>
size_t FindPosition(const SArray<K>& src, const SArray<K>& dst, SArray<int>* pos) {
  CHECK_NOTNULL(pos)->resize(dst.size());
  return FindPosition(src.begin(), src.end(),
                      dst.begin(), dst.end(),
                      pos->begin(), pos->end());
}
}  // namespace difacto
#endif  // DIFACTO_COMMON_FIND_POSITION_H_
