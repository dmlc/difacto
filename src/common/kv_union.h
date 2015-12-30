#ifndef DIFACTO_COMMON_KV_UNION_H_
#define DIFACTO_COMMON_KV_UNION_H_
#include <vector>
#include "./kv_match.h"
namespace difacto {


/**
 * \brief Join two key-value lists
 *
 * \code
 * key_a = {1,2,3};
 * val_a = {2,3,4};
 * key_b = {1,3,5};
 * val_b = {3,4,5};
 * KVUnion(key_a, val_a, key_b, val_b, &joined_key, &joined_val);
 * // then joined_key = {1,2,3,5}; and joined_val = {5,3,8,5};
 * \endcode
 *
 * \tparam K type of key
 * \tparam V type of value
 * @param keys_a keys from list a
 * @param vals_a values from list a
 * @param keys_b keys from list b
 * @param vals_b values from list b
 * @param joined_key the union of key1 and key2
 * @param joined_val the union of val1 and val2
 * @param val_len the length of a single value
 * @param op the assignment operator (default is PLUS)
 * @param num_threads number of thread (default is 2)
 */
template <typename K, typename V>
void KVUnion(
    const std::vector<K>& keys_a,
    const std::vector<V>& vals_a,
    const std::vector<K>& keys_b,
    const std::vector<V>& vals_b,
    std::vector<K>* joined_keys,
    std::vector<V>* joined_vals,
    int val_len = 1,
    AssignOp op = PLUS,
    int num_threads = DEFAULT_NTHREADS) {
  // merge keys
  CHECK_NOTNULL(joined_keys)->clear();
  std::set_union(keys_a.begin(), keys_a.end(), keys_b.begin(), keys_b.end(),
      std::back_inserter(*joined_keys));

  // merge value of list a
  CHECK_NOTNULL(joined_vals)->clear();
  size_t n1 = KVMatch<K, V>(
      keys_a, vals_a, *joined_keys, joined_vals, val_len, ASSIGN, num_threads);
  CHECK_EQ(n1, keys_a.size() * val_len);

  // merge value list b
  auto n2 = KVMatch<K, V>(
      keys_b, vals_b, *joined_keys, joined_vals, val_len, op, num_threads);
  CHECK_EQ(n2, keys_b.size() * val_len);
}

}  // namespace difacto
#endif  // DIFACTO_COMMON_KV_UNION_H_
