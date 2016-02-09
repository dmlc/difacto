#ifndef DIFACTO_DATA_DATA_STORE_IMPL_H_
#define DIFACTO_DATA_DATA_STORE_IMPL_H_
#include <queue>
#include <thread>
#include <mutex>
#include <string>
#include <functional>
#include <unordered_map>
#include "common/range.h"
#include "difacto/sarray.h"
namespace difacto {

class DataStoreImpl {
 public:
  DataStoreImpl() { }
  virtual ~DataStoreImpl() { }
  /**
   * \brief push a data into the store
   *
   * @param key the unique key
   * @param data the data buff
   */
  virtual void Store(const std::string& key, const SArray<char>& data) = 0;
  /**
   * \brief pull data from the store
   *
   * @param key the unique key
   * @param range only pull a range of the data. If it is Range::All(), then pul
   * the whole data
   * @param data the pulled data
   */
  virtual void Fetch(const std::string& key, Range range, SArray<char>* data) = 0;

  /**
   * \brief pretech a data
   *
   * @param key
   * @param range
   */
  virtual void Prefetch(const std::string& key, Range range) = 0;

  /**
   * \brief remove data from the store
   * \param key the unique key of the data
   */
  virtual void Remove(const std::string& key) = 0;
};

/**
 * \brief a naive implementation which puts all things in memory
 */
class DataStoreMemory : public DataStoreImpl {
 public:
  DataStoreMemory() { }
  virtual ~DataStoreMemory() { }
  void Store(const std::string& key, const SArray<char>& data) override {
    store_[key] = data;
  }
  void Fetch(const std::string& key, Range range, SArray<char>* data) override {
    auto it = store_.find(key);
    CHECK(it != store_.end());
    *CHECK_NOTNULL(data) = it->second.segment(range.begin, range.end);
  }
  void Prefetch(const std::string& key, Range range) override { }
  void Remove(const std::string& key) override { store_.erase(key); }
 private:
  std::unordered_map<std::string,SArray<char>> store_;
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
