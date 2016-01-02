#ifndef DIFACTO_DATA_DATA_STORE_H_
#define DIFACTO_DATA_DATA_STORE_H_
#include <utility>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "common/range.h"
#include "dmlc/data.h"
#include "./data_store_impl.h"
#include "./shared_row_block_container.h"
#include "difacto/sarray.h"
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
    DataType type;
    type.code = typeid(V).hash_code();
    type.size = sizeof(V);
    data_types_[key] = type;
    store_->Push(std::to_string(key), data);
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
  void Pull(int key, SArray<V>* data, Range range = Range::All()) {
    auto it = data_types_.find(key);
    CHECK(it != data_types_.end()) << "key " << key << " dosen't exist";
    CHECK_EQ(it->second.code, typeid(V).hash_code());
    SArray<char> cdata;
    store_->Pull(std::to_string(key), GetCharRange(key, range), &cdata);
    *CHECK_NOTNULL(data) = cdata;
  }

  /**
   * \brief give a hit to the store what will be pulled next.
   *
   * the store may use the hint to perform data pretech
   * @param key the unique key
   * @param range an optional range
   */
  virtual void NextPullHint(int key, Range range = Range::All()) {
    if (IsRowBlockKey(key)) {
      auto keys = GetRowBlockKeys(key);
      Range rg1 = range == Range::All() ? range
                  : Range(range.begin, range.end+1) * sizeof(size_t);
      store_->Prefetch(keys[0], rg1, [this, key, keys, range](const SArray<char>& data) {
          SArray<size_t> offset(data);
          Range rg3 = Range(offset[0], offset.back());
          store_->Prefetch(keys[3], GetCharRange(key, rg3));
          store_->Prefetch(keys[4], rg3 * sizeof(real_t));
        });
      Range rg2 = range == Range::All() ? range : range * sizeof(real_t);
      store_->Prefetch(keys[1], rg2);
      store_->Prefetch(keys[2], rg2);
    } else {
      store_->Prefetch(std::to_string(key), GetCharRange(key, range));
    }
  }

  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  virtual void Remove(int key) {
    if (IsRowBlockKey(key)) {
      for (const auto& s : GetRowBlockKeys(key)) {
        store_->Remove(s);
      }
    } else {
      store_->Remove(std::to_string(key));
    }
  }

  /**
   * \brief copy a rowblock into the store
   *
   * @param key the unique key
   * @param data the rowblock
   */
  template <typename T>
  void Push(int key, const dmlc::RowBlock<T>& data) {
    SharedRowBlockContainer<T> blk(data);
    Push(key, blk);
  }

  /**
   * \brief push a shared rowblock container into the store (no memory copy)
   *
   * @param key the unique key
   * @param data the rowblock container
   */
  template <typename T>
  void Push(int key, const SharedRowBlockContainer<T>& data) {
    DataType type;
    type.code = typeid(T).hash_code();
    type.size = sizeof(T);
    data_types_[key] = type;
    rowblk_keys_.insert(key);
    CHECK_EQ(data.offset[0], 0);
    auto keys = GetRowBlockKeys(key);
    store_->Push(keys[0], data.offset);
    store_->Push(keys[1], data.label);
    store_->Push(keys[2], data.weight);
    store_->Push(keys[3], data.index);
    store_->Push(keys[4], data.value);
  }

  /**
   * \brief pull rowblock from the store
   *
   * @param key the unique key
   * @param range an optional row range for pulling
   * @param data the pulled data
   */
  template <typename T>
  void Pull(int key, SharedRowBlockContainer<T>* data, Range range = Range::All()) {
    CHECK_NOTNULL(data);
    CHECK(IsRowBlockKey(key));
    auto keys = GetRowBlockKeys(key);
    Range rg1 = range == Range::All() ? range
                : Range(range.begin, range.end+1) * sizeof(size_t);
    store_->Pull(keys[0], rg1, &data->offset);
    Range rg2 = range == Range::All() ? range : range * sizeof(real_t);
    store_->Pull(keys[1], rg2, &data->label);
    store_->Pull(keys[2], rg2, &data->weight);
    Range rg3 = Range(data->offset[0], data->offset.back());
    store_->Pull(keys[3], rg3 * sizeof(T), &data->index);
    store_->Pull(keys[4], rg3 * sizeof(real_t), &data->value);
  }

 private:
  std::vector<std::string> GetRowBlockKeys(int key) {
    std::string skey = std::to_string(key) + "_";
    return {skey+"offset", skey+"label", skey+"weight", skey+"index", skey+"value"};
  }
  inline bool IsRowBlockKey(int key) { return rowblk_keys_.count(key) != 0; }
  inline Range GetCharRange(int key, Range range) {
    auto it = data_types_.find(key);
    CHECK(it != data_types_.end()) << "key " << key << " dosen't exist";
    return (range == Range::All() ? range : range * it->second.size);
  }
  std::unordered_set<int> rowblk_keys_;
  struct DataType { size_t code; size_t size; };
  std::unordered_map<int, DataType> data_types_;
  DataStoreImpl* store_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_H_
