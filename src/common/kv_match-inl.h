/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef _KV_MATCH_INL_H_
#define _KV_MATCH_INL_H_
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

  // drop the unmatched tail of src
  src_key = std::lower_bound(src_key, src_key_end, *dst_key);
  src_val += (src_key - (src_key_end - src_len)) * k;

  if (dst_len <= grainsize) {
    while (dst_key != dst_key_end && src_key != src_key_end) {
      if (*src_key < *dst_key) {
        ++ src_key; src_val += k;
      } else {
        if (!(*dst_key < *src_key)) {  // equal
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

}  // namespace difacto
#endif  // _KV_MATCH_INL_H_
