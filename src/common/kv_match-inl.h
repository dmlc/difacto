/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_KV_MATCH_INL_H_
#define DIFACTO_COMMON_KV_MATCH_INL_H_
namespace difacto {

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

  // drop the unmatched head of src
  src_key = std::lower_bound(src_key, src_key_end, *dst_key);
  src_val += (src_key - (src_key_end - src_len)) * k;

  if (dst_len <= grainsize) {
    while (dst_key != dst_key_end && src_key != src_key_end) {
      if (*src_key < *dst_key) {
        ++src_key; src_val += k;
      } else {
        if (!(*dst_key < *src_key)) {  // equal
          for (int i = 0; i < k; ++i) {
            AssignFunc(src_val[i], op, &dst_val[i]);
          }
          ++src_key; src_val += k;
          *n += k;
        }
        ++dst_key; dst_val += k;
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
        dst_key + dst_len / 2, dst_key_end, dst_val + (dst_len / 2) * k,
        k, op, grainsize, &m);
    thr.join();
    *n += m;
  }
}

/**
 * \brief thread function, internal use
 */
template <typename K, typename I, typename V>
void KVMatchVaryLen(
    const K* src_key,
    const K* src_key_end,
    const I* src_len,
    const V* src_val,
    const K* dst_key,
    const K* dst_key_end,
    const I* dst_len,
    V* dst_val,
    AssignOp op,
    size_t grainsize,
    size_t* n) {
  size_t src_size = std::distance(src_key, src_key_end);
  size_t dst_size = std::distance(dst_key, dst_key_end);
  if (dst_size == 0 || src_size == 0) return;

  // drop the unmatched head of src
  src_key = std::lower_bound(src_key, src_key_end, *dst_key);
  size_t k = (src_key - (src_key_end - src_size));
  for (size_t i = 0; i < k; ++i) src_val += src_len[i];
  src_len += k;

  if (dst_size <= grainsize) {
    while (dst_key != dst_key_end && src_key != src_key_end) {
      if (*src_key < *dst_key) {
        ++src_key; src_val += *src_len; ++src_len;
      } else {
        if (!(*dst_key < *src_key)) {  // equal
          I k = *src_len;
          CHECK_EQ(k, *dst_len);
          for (I i = 0; i < k; ++i) {
            AssignFunc(src_val[i], op, &dst_val[i]);
          }
          ++src_key; src_val += k; ++src_len;
          *n += k;
        }
        ++dst_key; dst_val += *dst_len; ++dst_len;
      }
    }
  } else {
    size_t step = dst_size / 2;
    std::thread thr(
        KVMatchVaryLen<K, I, V>, src_key, src_key_end, src_len, src_val,
        dst_key, dst_key + step, dst_len, dst_val,
        op, grainsize, n);
    for (size_t i = 0; i < step; ++i) dst_val += dst_len[i];
    size_t m = 0;
    KVMatchVaryLen<K, I, V>(
        src_key, src_key_end, src_len, src_val,
        dst_key + step, dst_key_end, dst_len + step, dst_val,
        op, grainsize, &m);
    thr.join();
    *n += m;
  }
}

}  // namespace difacto
#endif  // DIFACTO_COMMON_KV_MATCH_INL_H_
