#ifndef DIFACTO_DATA_DATA_CACHE_H_
#define DIFACTO_DATA_DATA_CACHE_H_
#include <queue>
#include <thread>
#include <mutex>
#include <utility>
#include "common/range.h"
#include "dmlc/data.h"
#include "./data_store_impl.h"
namespace difacto {

/**
 * \brief data store can be used to store and fetch data. Once the pushed data
 * exceed the maximal memory cacapcity, it will dump data into disks
 */
class DataStore {
 public:
  /**
   * \brief create a data store
   *
   * @param store_prefix the prefix to store the data store, such as /tmp/store_
   * @param max_mem_capacity the maximal memory it can used
   */
  DataStore(const std::string& store_prefix, size_t max_mem_capacity) {
    store_ = new DataStoreMemory();
  }

  ~DataStore() { delete store_; }

  /**
   * \brief push data into the store
   *
   * @param key the unique key
   * @param data the data buffer
   * @param size the data size
   */
  template <typename V>
  void Push(int key, const V* data, size_t size) {
    Push_(key * kOS, data, size);
  }

  /**
   * \brief push a rowblock into the store
   *
   * @param key the unique key
   * @param data the rowblock
   */
  template <typename T>
  void Push(int key, dmlc::RowBlock<T> data) {
    CHECK_EQ(data.offset[0], 0);
    Push_(key * kOS, data.offset, data.size+1);
    if (data.label != nullptr) {
      Push_(key * kOS + 1, data.label, data.size);
    }
    if (data.weight != nullptr) {
      Push_(key * kOS + 2, data.weight, data.size);
    }
    size_t nnz = data.offset[data.size] - data.offset[0];
    Push_(key * kOS + 3, data.index, nnz);
    if (data.value != nullptr) {
      Push_(key * kOS + 4, data.value, nnz);
    }
  }

  /**
   * \brief pull data from the store
   *
   * @param key the unique key
   * @param data the pulled data buffer
   * @param range an optional range for pulling
   *
   * \code
   * int data[] = {0,1,2,3};
   * Push(0, data, 4);
   * int *ret;
   * EXPECT_EQ(2, Pull(0, &ret, Range(1,3)));
   * EXPECT_EQ(ret[0], 1);
   * EXPECT_EQ(ret[1], 2);
   * \endcode
   * @return the data size
   */
  template<typename V>
  size_t Pull(int key, V** data, Range range = Range::All()) {
    size_t size;
    Pull_(key * kOS, range, false, data, &size);
    return size;
  }

  /**
   * \brief pull rowblock from the store
   *
   * @param key the unique key
   * @param data the pulled rowblock buff
   * @param range an optional row range for pulling
   */
  template <typename T>
  void Pull(int key, dmlc::RowBlock<T>* data, Range range = Range::All()) {
    dmlc::RowBlock<T> out;
    out.size = Pull_(key * kOS, range, false, &out.offset);
    Pull_(key * kOS + 1, range, true, &out.label);
    Pull_(key * kOS + 2, range, true, &out.weight);
    Range rg2 = Range(out.offset[0], out.offset[out.size]);
    Pull_(key * kOS + 3, rg2, false, &out.index);
    Pull_(key * kOS + 4, rg2, true, &out.value);
    *CHECK_NOTNULL(data) = out;
  }


  /**
   * \brief give a hit to the store what will be pulled next.
   *
   * the store may use the hint to perform data pretech
   * @param key the unique key
   * @param range an optional range
   */
  void NextPullHint(int key, Range range = Range::All()) {
    store_->NextPullHint(key, range);
  }

 private:
  static const int kOS = 10;

  template<typename V>
  void Push_(int key, const V* data, size_t size) {
    store_->Push(key, static_cast<char*>(data), size * sizeof(V),
                 typeid(V).hash_code());
  }

  template<typename V>
  size_t Pull_(int key, Range range, bool allow_nonexist, V** data) {
    char** val;
    size_t ret = store_->Pull(
        key, range, typeid(V).hash_code(), allow_nonexist, val);
    *CHECK_NOTNULL(data) = static_cast<V*>(*val);
    return ret / sizeof(V);
  }

  DataStoreImpl* store_;
};

}  // namespace difacto
#endif /* DIFACTO_DATA_DATA_STORE_H_ */
