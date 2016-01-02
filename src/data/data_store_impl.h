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
  DataStoreImpl();
  virtual ~DataStoreImpl();
  /**
   * \brief push a data into the store
   *
   * @param key the unique key
   * @param data the data buff
   */
  virtual void Push(const std::string& key, const SArray<char>& data) = 0;
  /**
   * \brief pull data from the store
   *
   * @param key the unique key
   * @param range only pull a range of the data. If it is Range::All(), then pul
   * the whole data
   * @param data the pulled data
   */
  virtual void Pull(const std::string& key, Range range, SArray<char>* data) = 0;

  typedef std::function<void(const SArray<char>& data)> Callback;
  /**
   * \brief pretech a data
   *
   * @param key
   * @param range
   * @param on_complete the callback when prefetch is done
   */
  virtual void Prefetch(const std::string& key, Range range,
                        Callback on_complete = nullptr) = 0;
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

  void Push(const std::string& key, const SArray<char>& data) override {
    store_[key] = data;
  }

  void Pull(const std::string& key, Range range, SArray<char>* data) override {
    auto it = store_.find(key);
    if (it == store_.end()) {
      *CHECK_NOTNULL(data) = SArray<char>();
    } else {
      if (range == Range::All()) {
        *CHECK_NOTNULL(data) = it->second;
      } else {
        *CHECK_NOTNULL(data) = it->second.segment(range.begin, range.end);
      }
    }
  }
  void Prefetch(const std::string& key, Range range, Callback on_complete) override {
    if (on_complete) {
      SArray<char> data;
      Pull(key, range, &data);
      on_complete(data);
    }
  }

  void Remove(const std::string& key) override {
    store_.erase(key);
  }
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
