#ifndef DIFACTO_DATA_DATA_STORE_H_
#define DIFACTO_DATA_DATA_STORE_H_
#include <utility>
#include <memory>
#include <vector>
#include <unordered_map>
#include "common/range.h"
#include "dmlc/io.h"
#include "./data_store_impl.h"
#include "difacto/sarray.h"
namespace difacto {
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
   * @param store_prefix , such as /tmp/store_. If not specified, then keep all
   * things in memory
   * @param max_mem_capacity  in default no limits
   */
  DataStore() { store_ = new DataStoreMemory(); }
  /** \brief deconstructor */
  virtual ~DataStore() { delete store_; }
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
    DataMeta meta;
    meta.type_code = typeid(V).hash_code();
    meta.type_size = sizeof(V);
    meta.data_size = data.size();
    data_meta_[key] = meta;
    store_->Store(key, SArray<char>(data));
  }
  /**
   * \brief pull data from the store
   *
   * \type_code
   * SArray<int> data = {0,1,2,3}, ret;
   * Store("data", data);
   * Fetch("data", &ret, Range(1,3));
   * EXPECT_EQ(ret[0], 1);
   * EXPECT_EQ(ret[0], 2);
   * \endtype_code
   *
   * @param key the unique key
   * @param data the feteched data
   * @param range an optional range for fetching
   */
  template <typename V>
  void Fetch(const std::string& key, SArray<V>* data, Range range = Range::All()) {
    auto char_range = GetCharRange(key, range);
    CHECK_EQ(data_meta_[key].type_code, typeid(V).hash_code());
    SArray<char> char_data;
    store_->Fetch(key, char_range, &char_data);
    *CHECK_NOTNULL(data) = char_data;
  }
  /**
   * \brief give a hit to the store what will be fetched in a near future.
   *
   * the store may use the hint to perform data pretech
   *
   * @param key the unique key
   * @param range an optional range
   */
  void Prefetch(const std::string& key, Range range = Range::All()) {
    store_->Prefetch(key, GetCharRange(key, range));
  }
  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  void Remove(const std::string& key) { store_->Remove(key); }

  /** \brief load meta data */
  void Load(dmlc::Stream *fi) {
    dmlc::istream is(fi);
    std::string header;
    is >> header;
    CHECK_EQ(header, meta_header_) << "invalid meta header";
    size_t size; is >> size;
    for (size_t i = 0; i < size; ++i) {
      std::string key; is >> key;
      DataMeta meta; is >> meta.data_size >> meta.type_code >> meta.type_size;
      data_meta_[key] = meta;
    }
  }

  /** \brief save meta data */
  void Save(dmlc::Stream *fo) const {
    dmlc::ostream os(fo);
    os << meta_header_ << "\t";
    os << data_meta_.size() << "\n";
    for (const auto it : data_meta_) {
      os << it.first << "\t" << it.second.data_size << "\t"
         << it.second.type_code << "\t" << it.second.type_size << "\n";
    }
  }

  /**
   * \brief return the data size of a key
   **/
  size_t size(const std::string& key) const {
    auto it = data_meta_.find(key);
    CHECK(it != data_meta_.end()) << "key " << key << " dosen't exist";
    return it->second.data_size;
  }

 private:
  inline Range GetCharRange(const std::string& key, Range range) {
    CHECK(range.Valid());
    size_t siz = size(key);
    if (range == Range::All()) range = Range(0, siz);
    CHECK_LE(range.end, siz);
    return range * data_meta_[key].type_size;
  }

  struct DataMeta {
    /** \brief data size */
    size_t data_size;
    /** \brief type type_code */
    size_t type_code;
    /** \brief sizeof(type) */
    size_t type_size;
  };
  std::unordered_map<std::string, DataMeta> data_meta_;
  DataStoreImpl* store_;
  const std::string meta_header_ = "data_store_meta(key,value_size,type_code,type_size)";

};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_H_
