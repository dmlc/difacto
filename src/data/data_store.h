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
    capacity_ = 0;
    store_prefix_ = store_dir;
    max_mem_cap_ = max_mem_capacity;
  }

  ~DataStore() {}

  template <typename V>
  void Push(int key, const V* data, size_t size) {
    Push_(key * kOS, data, size);
  }

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


  template<typename V>
  size_t Pull(int key, V** data, Range range = Range::All()) {
    size_t size;
    Pull_(key * kOS, range, false, data, &size);
    return size;
  }

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

  void AddPretechHit(int key, Range range) {
    std::lock_guard<std::mutex> lk(mu_);
    pretch_list_.push(std::make_pair(key, range));
  }

 private:
  static const int kOS = 10;

  template<typename V>
  void Push_(int key, const V* data, size_t len) {

  }

  template<typename V>
  size_t Pull_(int key, Range range, bool allow_null, V** data) {

  }
  std::queue<std::pair<int, Range >> pretch_list_;
  std::mutex mu_;
  std::string store_prefix_;
  size_t max_mem_cap_;
  size_t capacity_;

  std::thread data_pretch_;
  struct DataEntry {
    int key;
    void* data;
    size_t size;
    size_t pos;
  };



};

/////////////////////// implementation //////////////////////


}  // namespace difacto
#endif /* DIFACTO_DATA_DATA_STORE_H_ */
