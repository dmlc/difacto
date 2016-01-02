#ifndef DIFACTO_COMMON_KV_MATCH_H_
#define DIFACTO_COMMON_KV_MATCH_H_
#include <vector>
#include <thread>
#include "dmlc/logging.h"
#include "./range.h"
#include "difacto/base.h"
namespace difacto {

/**
 * \brief assignment operator
 */
enum AssignOp {
  ASSIGN,  // a = b
  PLUS,    // a += b
  MINUS,   // a -= b
  TIMES,   // a *= b
  DIVIDE,  // a -= b
  AND,     // a &= b
  OR,      // a |= b
  XOR      // a ^= b
};

/**
 * \brief return an assignment function: right op= left
 */
template<typename T>
inline void AssignFunc(const T& lhs, AssignOp op, T* rhs) {
  switch (op) {
    case ASSIGN:
      *rhs = lhs; break;
    case PLUS:
      *rhs += lhs; break;
    case MINUS:
      *rhs -= lhs; break;
    case TIMES:
      *rhs *= lhs; break;
    case DIVIDE:
      *rhs /= lhs; break;
    default:
      LOG(FATAL) << "use AssignOpInt.." ;
  }
}

namespace {
/**
 * \brief thread function, internal use
 *
 * \param src_key start of source key
 * \param src_key_end end of source key
 * \param src_val start of source val
 * \param dst_key start of destination key
 * \param dst_key_end end of denstination key
 * \param dst_val start of destination val
 * \param k length of a single value
 * \param op assignment operator
 * \param grainsize thread grainsize size
 * \param n number of matched kv pairs
 */
template <typename K, typename V>
void KVMatch(
    const K* src_key, const K* src_key_end, const V* src_val,
    const K* dst_key, const K* dst_key_end, V* dst_val,
    int k, AssignOp op, size_t grainsize, size_t* n) {
  size_t src_len = std::distance(src_key, src_key_end);
  size_t dst_len = std::distance(dst_key, dst_key_end);
  if (dst_len == 0 || src_len == 0) return;

  // drop the unmatched tail of src
  src_key = std::lower_bound(src_key, src_key_end, *dst_key);
  src_val += (src_key - (src_key_end - src_len)) * k;

  if (dst_len <= grainsize) {
    while (dst_key != dst_key_end && src_key != src_key_end) {
      if (*src_key < *dst_key) {
        ++ src_key; src_val += k;
      } else {
        if (!(*dst_key < *src_key)) {
          for (int i = 0; i < k; ++i) {
            AssignFunc(src_val[i], op, &dst_val[i]);
          }
          ++ src_key; src_val += k;
          *n += k;
        }
        ++ dst_key; dst_val += k;
      }
    }
  } else {
    std::thread thr(
        KVMatch<K, V>, src_key, src_key_end, src_val,
        dst_key, dst_key + dst_len / 2, dst_val,
        k, op, grainsize, n);
    size_t m = 0;
    KVMatch<K, V>(
        src_key, src_key_end, src_val,
        dst_key + dst_len / 2, dst_key_end, dst_val + ( dst_len / 2 ) * k,
        k, op, grainsize, &m);
    thr.join();
    *n += m;
  }
}


/**
 * \brief Find the index range of a segment of a sorted array such that the
 * entries in this segment is within [lower, upper). Assume
 * array values are ordered.
 *
 * An example
 * \code{cpp}
 * SArray<int> a{1 3 5 7 9};
 * CHECK_EQ(Range(1,3), FindRange(a, 2, 7);
 * \endcode
 *
 * \param arr the source array
 * \param lower the lower bound
 * \param upper the upper bound
 *
 * \return the index range
 */
template<typename V>
Range FindRange(const std::vector<V>& arr, V lower, V upper) {
  if (upper <= lower) return Range(0,0);
  auto lb = std::lower_bound(arr.begin(), arr.end(), lower);
  auto ub = std::lower_bound(arr.begin(), arr.end(), upper);
  return Range(lb - arr.begin(), ub - arr.begin());
}

}  // namespace


/**
 * \brief Merge \a src_val into \a dst_val by matching keys. Keys must be unique
 * and sorted.
 *
 * \code
 * if (dst_key[i] == src_key[j]) {
 *    dst_val[i] op= src_val[j]
 * }
 * \endcode
 *
 * \code
 * src_key = {1,2,3};
 * src_val = {6,7,8};
 * dst_key = {1,3,5};
 * KVMatch(src_key, src_val, dst_key, &dst_val);
 * // then dst_val = {6,8,0};
 * \endcode
 * When finished, \a dst_val will have length `k * dst_key.size()` and filled
 * with matched value. Umatched value will be untouched if exists or filled with 0.
 *
 * \tparam K type of key
 * \tparam V type of value
 * \param src_key the source keys
 * \param src_val the source values
 * \param dst_key the destination keys
 * \param dst_val the destination values.
 * \param val_len the length of a single value (default is 1)
 * \param op the assignment operator (default is ASSIGN)
 * \param num_threads number of thread (default is 2)
 * \return the number of matched kv pairs
 */
template <typename K, typename V>
size_t KVMatch(
    const std::vector<K>& src_key,
    const std::vector<V>& src_val,
    const std::vector<K>& dst_key,
    std::vector<V>* dst_val,
    int val_len = 1,
    AssignOp op = ASSIGN,
    int num_threads = DEFAULT_NTHREADS) {
  // do check
  CHECK_GT(num_threads, 0);
  CHECK_EQ(src_key.size() * val_len, src_val.size());
  CHECK_NOTNULL(dst_val)->resize(dst_key.size() * val_len);
  if (dst_key.empty()) return 0;

  // shorten the matching range
  Range range = FindRange(dst_key, src_key.front(), src_key.back()+1);
  size_t grainsize = std::max(range.Size() * val_len / num_threads + 5,
                              (size_t)1024*1024);
  size_t n = 0;
  KVMatch<K, V>(
      src_key.data(), src_key.data() + src_key.size(), src_val.data(),
      dst_key.data() + range.begin, dst_key.data() + range.end,
      dst_val->data() + range.begin * val_len, val_len, op, grainsize, &n);
  return n;
}


// template <typename K, typename V>
// size_t KVMatch(
//    const K* src_key_begin, const K* src_key_end,
//    const V* src_val_begin, const V* src_val_end,
//    const K* dst_key_begin, const K* dst_key_end,
//    V* dst_val_begin,
//    int val_len = 1,
//    AssignOp op = ASSIGN,
//    int num_threads = DEFAULT_NTHREADS) {
//   // TODO
// }


}  // namespace difacto
#endif  // DIFACTO_COMMON_KV_MATCH_H_
