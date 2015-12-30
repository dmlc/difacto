#ifndef DIFACTO_DATA_DATA_STORE_IMPL_H_
#define DIFACTO_DATA_DATA_STORE_IMPL_H_
#include <unordered_map>
#include "common/range.h"
namespace difacto {

class DataStoreImpl {
 public:
  DataStoreImpl();
  virtual ~DataStoreImpl();

  /**
   * \brief push a data into the store
   *
   * @param key the unique key
   * @param data the data buff
   * @param size the data size
   * @param type the data type
   */
  virtual void Push(int key, const char* data, size_t size, size_t type) = 0;

  /**
   * \brief pull data from the store
   *
   * @param key the unique key
   * @param range only pull a range of the data. If it is Range::All(), then pul
   * the whole data
   * @param type the data type
   * @param allow_nonexist if true, then set data to NULL if the key doesn't exist
   * @param data the pulled data buff
   *
   * @return the size of data
   */
  virtual size_t Pull(int key, Range range, size_t type, bool allow_nonexist,
                      char** data) = 0;

  /**
   * \brief add a hint to tell the data store do pretech
   */
  virtual void NextPullHint(int key, Range range) {}


  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  virtual void Remove(int key) = 0;

};

/**
 * \brief a naive implementation which puts all things in memory
 */
class DataStoreMemory : public DataStoreImpl {
 public:
  DataStoreMemory() { }
  virtual ~DataStoreMemory() { }

  void Push(int key, const char* data, size_t size, size_t type) override {
    CHECK(store_.count(key) == 0) << "duplicate key: " << key;
    auto& entry = store_[key];
    entry.type = type;
    entry.data.resize(size);
    memcpy(entry.data.data(), data, size);
  }

  size_t Pull(int key, Range range, size_t type, bool allow_nonexist,
              char** data) override {
    auto it = store_.find(key);
    if (it == store_.end()) {
      CHECK(allow_nonexist) << "key " << key << " does not exist";
      *data = nullptr;
      return 0;
    } else {
      CHECK_EQ(type, it->second.type);
      CHECK(range.Valid());
      *data = it->second.data.data() + range.begin;
      if (range == Range::All()) {
        return it->second.data.size();
      } else {
        CHECK_LE(range.end, it->second.data.size());
        return range.Size();
      }
    }
  }

  void Remove(int key) override {
    store_.erase(key);
  }
 private:
  struct DataEntry {
    std::vector<char> data;
    size_t type;
  };
  std::unordered_map<int, DataEntry> store_;
};

/**
 * \brief write data back to disk if exeeds the maximal memory capacity
 */
class DataStoreDisk : public DataStoreImpl {
 public:

  DataStoreDisk(const std::string& cache_prefix,
                size_t max_mem_capacity) {
  }
  virtual ~DataStoreDisk() { }

};

}  // namespace difacto
#endif  // DIFACTO_DATA_DATA_STORE_IMPL_H_
