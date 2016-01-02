#ifndef DIFACTO_DATA_DATA_CACHE_H_
#define DIFACTO_DATA_DATA_CACHE_H_
#include <utility>
#include <memory>
#include <vector>
#include "common/range.h"
#include "dmlc/data.h"
#include "./data_store_impl.h"
#include "./shared_row_block_container.h"
#include "ps/sarray.h"
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
   * \brief copy data into the store, overwrite the previous data if key exists.
   *
   * @param key the unique key
   * @param data the data buffer
   * @param size the data size
   */
  template <typename V>
  void Push(int key, const V* data, size_t size) {
    ps::SArray<V> sdata; sdata.CopyFrom(data, size);
    Push(key, sdata);
  }

  /**
   * \brief push data into the store. no data copy. overwrite the previous data
   * if key exists.
   *
   * @param key the unique key
   * @param data the data
   */
  template <typename V>
  void Push(int key, const std::shared_ptr<std::vector<V>>& data) {
    ps::SArray<V> sdata(data);
    Push(key, sdata);
  }

  /**
   * \brief push data into the store.no data copy. overwrite the previous data
   * if key exists.
   *
   * @param key the unique key
   * @param data the data
   */
  template <typename V>
  void Push(int key, const SArray<V>& data) {
    // TODO
  }

  /**
   * \brief pull data from the store
   *
   * \code
   * int data[] = {0,1,2,3};
   * Push(0, data, 4);
   * auto ret = Pull(Range(1,3));
   * EXPECT_EQ(ret[0], 1);
   * EXPECT_EQ(ret[0], 2);
   * \endcode
   *
   * @param key the unique key
   * @param range an optional range for pulling
   *
   * @return the data
   */
  template <typename V>
  SArray<V> Pull(int key, Range range = Range::All()) {
    // TODO
  }


  /**
   * \brief give a hit to the store what will be pulled next.
   *
   * the store may use the hint to perform data pretech
   * @param key the unique key
   * @param range an optional range
   */
  virtual void NextPullHint(int key, Range range = Range::All()) {
    store_->NextPullHint(key, range);
  }

  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  virtual void Remove(int key) {
    store_->Remove(key);
  }

  /**
   * \brief copy a rowblock into the store
   *
   * @param key the unique key
   * @param data the rowblock
   */
  template <typename T>
  void Push(int key, dmlc::RowBlock<T> data) {
    CHECK_EQ(data.offset[0], 0);
    SharedRowBlockContainer<T> blk;
    blk.offset.CopyFrom(data.offset, data.size+1);
    if (data.label != nullptr) {
      blk.label.CopyFrom(data.label, data.size);
    }
    if (data.weight != nullptr) {
      blk.weight.CopyFrom(data.weight, data.size);
    }
    size_t nnz = data.offset[data.size] - data.offset[0];
    blk.index.CopyFrom(data.index, nnz);
    if (data.value != nullptr) {
      blk.value.CopyFrom(data.value, nnz);
    }
    Push(blk);
  }

  /**
   * \brief push a shared rowblock container into the store (no memory copy)
   *
   * @param key the unique key
   * @param data the rowblock container
   */
  template <typename T>
  void Push(int key, const SharedRowBlockContainer<T>& data) {

  }

  /**
   * \brief pull rowblock from the store
   *
   * @param key the unique key
   * @param range an optional row range for pulling
   */
  template <typename T>
  SharedRowBlockContainer<T> Pull(int key, Range range = Range::All()) {
    // dmlc::RowBlock<T> out;
    // out.size = Pull_(key * kOS, range, false, &out.offset);
    // Pull_(key * kOS + 1, range, true, &out.label);
    // Pull_(key * kOS + 2, range, true, &out.weight);
    // Range rg2 = Range(out.offset[0], out.offset[out.size]);
    // Pull_(key * kOS + 3, rg2, false, &out.index);
    // Pull_(key * kOS + 4, rg2, true, &out.value);
    // *CHECK_NOTNULL(data) = out;
    SharedRowBlockContainer<T> block;
    return block;
  }
 protected:
  static const int kOS = 10;


  // template<typename V>
  // void Push_(int key, const V* data, size_t size) {
  //   store_->Push(key, reinterpret_cast<const char*>(data), size * sizeof(V),
  //                typeid(V).hash_code());
  // }

  template<typename V>
  size_t Pull_(int key, Range range, bool allow_nonexist, V** data) {
    char** val;
    size_t ret = store_->Pull(
        key, range, typeid(V).hash_code(), allow_nonexist, val);
    *CHECK_NOTNULL(data) = reinterpret_cast<V*>(*val);
    return ret / sizeof(V);
  }

  DataStoreImpl* store_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_H_
