#ifndef DIFACTO_DATA_DATA_STORE_H_
#define DIFACTO_DATA_DATA_STORE_H_
#include <utility>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "common/range.h"
#include "dmlc/data.h"
#include "dmlc/parameter.h"
#include "./data_store_impl.h"
#include "./shared_row_block_container.h"
#include "difacto/sarray.h"
namespace difacto {

/** \brief parameters for data store */
struct DataStoreParam : public dmlc::Parameter<DataStoreParam> {
  /** \brief the prefix to store the data cache */
  std::string data_cache_prefix;
  /** \brief the maximal memory data store can used. in default no limits */
  std::string max_data_store_memory;
  DMLC_DECLARE_PARAMETER(DataStoreParam) {
    DMLC_DECLARE_FIELD(data_cache_prefix);
    DMLC_DECLARE_FIELD(max_data_store_memory);
  }
};

/**
 * \brief data store can be used to store and fetch data. Once the stored data
 * exceed the maximal memory cacapcity, it will dump data into disks
 *
 * Instead of using Push and Pull, we use \ref Store and \ref Fetch here to
 * differential that here all functions are synchronous. To improve the
 * performance, we can use \ref Fefetch before a Fetch.
 */
class DataStore {
 public:
  /**
   * \brief create a data store
   *
   * @param store_prefix , such as
   * /tmp/store_. If not specified, then keep all things in memory
   * @param max_mem_capacity  in default no limits
   */
  DataStore() { store_ = new DataStoreMemory(); }
  /** \brief deconstructor */
  virtual ~DataStore() { delete store_; }
  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  KWArgs Init(const KWArgs& kwargs) {
    return kwargs;
  }
  /**
   * \brief copy data into the store, overwrite the previous data if key exists.
   *
   * @param key the unique key
   * @param data the data buffer
   * @param size the data size
   */
  template <typename V>
  void Store(const std::string& key, const V* data, size_t size) {
    ps::SArray<V> sdata; sdata.CopyFrom(data, size);
    Store(key, sdata);
  }
  /**
   * \brief store data without data copy. overwrite the previous data
   * if key exists.
   *
   * @param key the unique key
   * @param data the data
   */
  template <typename V>
  void Store(const std::string& key, const SArray<V>& data) {
    DataType type;
    type.code = typeid(V).hash_code();
    type.size = sizeof(V);
    data_types_[key] = type;
    store_->Store(key, SArray<char>(data));
  }
  /**
   * \brief pull data from the store
   *
   * \code
   * int data[] = {0,1,2,3};
   * Store(0, data, 4);
   * auto ret = Fetch(Range(1,3));
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
  void Fetch(const std::string& key, SArray<V>* data, Range range = Range::All()) {
    auto char_range = GetCharRange(key, range);
    CHECK_EQ(data_types_[key].code, typeid(V).hash_code());
    SArray<char> char_data;
    store_->Fetch(key, char_range, &char_data);
    *CHECK_NOTNULL(data) = char_data;
  }
  /**
   * \brief give a hit to the store what will be pulled next.
   *
   * the store may use the hint to perform data pretech
   * @param key the unique key
   * @param range an optional range
   */
  virtual void Prefetch(const std::string& key, Range range = Range::All()) {
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
      store_->Prefetch(key, GetCharRange(key, range));
    }
  }
  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  virtual void Remove(const std::string& key) {
    if (IsRowBlockKey(key)) {
      for (const auto& s : GetRowBlockKeys(key)) {
        store_->Remove(s);
      }
    } else {
      store_->Remove(key);
    }
  }
  /**
   * \brief copy a rowblock into the store
   *
   * @param key the unique key
   * @param data the rowblock
   */
  template <typename T>
  void Store(const std::string& key, const dmlc::RowBlock<T>& data) {
    SharedRowBlockContainer<T> blk(data);
    Store(key, blk);
  }
  /**
   * \brief push a shared rowblock container into the store (no memory copy)
   *
   * @param key the unique key
   * @param data the rowblock container
   */
  template <typename T>
  void Store(const std::string& key, const SharedRowBlockContainer<T>& data) {
    rowblk_keys_.insert(key);
    CHECK_EQ(data.offset[0], 0);
    auto keys = GetRowBlockKeys(key);
    Store(keys[0], data.offset);
    Store(keys[1], data.label);
    Store(keys[2], data.weight);
    Store(keys[3], data.index);
    Store(keys[4], data.value);
  }
  /**
   * \brief pull rowblock from the store
   *
   * @param key the unique key
   * @param range an optional row range for pulling
   * @param data the pulled data
   */
  template <typename T>
  void Fetch(const std::string& key, SharedRowBlockContainer<T>* data,
            Range range = Range::All()) {
    CHECK_NOTNULL(data);
    CHECK(IsRowBlockKey(key));
    auto keys = GetRowBlockKeys(key);
    Range rg1 = range == Range::All() ? range
                : Range(range.begin, range.end+1);
    Fetch(keys[0], &data->offset, rg1);
    Fetch(keys[1], &data->label, range);
    Fetch(keys[2], &data->weight, range);
    Range rg3 = Range(data->offset[0], data->offset.back());
    Fetch(keys[3], &data->index, rg3);
    Fetch(keys[4], &data->value, rg3);
    if (rg3.begin != 0) {
      SArray<size_t> offset; offset.CopyFrom(data->offset);
      for (size_t& o : offset) o -= rg3.begin;
      data->offset = offset;
    }

  }

 private:
  std::vector<std::string> GetRowBlockKeys(const std::string& key) {
    return {key+"_offset", key+"_label", key+"_weight", key+"_index", key+"_value"};
  }

  inline bool IsRowBlockKey(const std::string& key) {
    return rowblk_keys_.count(key) != 0;
  }

  inline Range GetCharRange(const std::string& key, Range range) {
    auto it = data_types_.find(key);
    CHECK(it != data_types_.end()) << "key " << key << " dosen't exist";
    return (range == Range::All() ? range : range * it->second.size);
  }
  std::unordered_set<std::string> rowblk_keys_;
  struct DataType { size_t code; size_t size; };
  std::unordered_map<std::string, DataType> data_types_;
  DataStoreImpl* store_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_H_
