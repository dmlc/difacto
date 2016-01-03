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
   * @param store_prefix the prefix to store the data store, such as
   * /tmp/store_. If not specified, then keep all things in memory
   * @param max_mem_capacity the maximal memory it can used. in default no limits
   */
  DataStore(const std::string& store_prefix = "", size_t max_mem_capacity = -1) {
    store_ = new DataStoreMemory();
  }
  /** \brief deconstructor */
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

  // /**
  //  * \brief push data into the store. no data copy. overwrite the previous data
  //  * if key exists.
  //  *
  //  * @param key the unique key
  //  * @param data the data
  //  */
  // template <typename V>
  // void Push(int key, const std::shared_ptr<std::vector<V>>& data) {
  //   ps::SArray<V> sdata(data);
  //   Push(key, sdata);
  // }

  /**
   * \brief push data into the store.no data copy. overwrite the previous data
   * if key exists.
   *
   * @param key the unique key
   * @param data the data
   */
  template <typename V>
  void Push(int key, const SArray<V>& data) {
    Push_(std::to_string(key), data);
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
    Pull_(std::to_string(key), range, data);
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
          Range rg = Range(offset.front(), offset.back());
          store_->Prefetch(keys[3], GetCharRange(keys[3], rg));
          store_->Prefetch(keys[4], GetCharRange(keys[4], rg));
        });
      store_->Prefetch(keys[1], GetCharRange(keys[1], range));
      store_->Prefetch(keys[2], GetCharRange(keys[2], range));
    } else {
      auto skey = std::to_string(key);
      store_->Prefetch(skey, GetCharRange(skey, range));
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
    rowblk_keys_.insert(key);
    CHECK_EQ(data.offset[0], 0);
    auto keys = GetRowBlockKeys(key);
    Push_(keys[0], data.offset);
    Push_(keys[1], data.label);
    Push_(keys[2], data.weight);
    Push_(keys[3], data.index);
    Push_(keys[4], data.value);
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
                : Range(range.begin, range.end+1);
    Pull_(keys[0], rg1, &data->offset);
    Pull_(keys[1], range, &data->label);
    Pull_(keys[2], range, &data->weight);
    Range rg3 = Range(data->offset[0], data->offset.back());
    Pull_(keys[3], rg3, &data->index);
    Pull_(keys[4], rg3, &data->value);
  }

 private:
  template <typename V>
  void Push_(const std::string& key, const SArray<V>& data) {
    DataType type;
    type.code = typeid(V).hash_code();
    type.size = sizeof(V);
    data_types_[key] = type;
    store_->Push(key, SArray<char>(data));
  }
  template <typename V>
  void Pull_(const std::string& key, Range range, SArray<V>* data) {
    auto char_range = GetCharRange(key, range);
    CHECK_EQ(data_types_[key].code, typeid(V).hash_code());
    SArray<char> char_data;
    store_->Pull(key, char_range, &char_data);
    *CHECK_NOTNULL(data) = char_data;
  }

  std::vector<std::string> GetRowBlockKeys(int key) {
    std::string skey = std::to_string(key) + "_";
    return {skey+"offset", skey+"label", skey+"weight", skey+"index", skey+"value"};
  }

  inline bool IsRowBlockKey(int key) { return rowblk_keys_.count(key) != 0; }

  inline Range GetCharRange(const std::string& key, Range range) {
    auto it = data_types_.find(key);
    CHECK(it != data_types_.end()) << "key " << key << " dosen't exist";
    return (range == Range::All() ? range : range * it->second.size);
  }
  std::unordered_set<int> rowblk_keys_;
  struct DataType { size_t code; size_t size; };
  std::unordered_map<std::string, DataType> data_types_;
  DataStoreImpl* store_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_H_
