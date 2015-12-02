#ifndef DIFACTO_PARALLEL_SORT_H_
#define DIFACTO_PARALLEL_SORT_H_
#include <thread>
#include <vector>
#include <algorithm>
namespace difacto {
namespace {
/**
 * \brief the thread function
 */
template<typename T, class Fn>
void ParallelSort_(T* data, size_t len, size_t grainsize, const Fn& cmp) {
  if (len <= grainsize) {
    std::sort(data, data + len, cmp);
  } else {
    std::thread thr(ParallelSort_<T, Fn>, data, len/2, grainsize, cmp);
    ParallelSort_(data + len/2, len - len/2, grainsize, cmp);
    thr.join();
    std::inplace_merge(data, data + len/2, data + len, cmp);
  }
}
}  // namespace

/**
 * @brief Parallel Sort
 *
 * @param arr the array for sorting
 * @param num_threads
 * @param cmp the comparision function, such as [](const T& a, const T& b) {
 * return a < b; } or an even simplier version: std::less<T>()
 */
template<typename T, class Fn>
void ParallelSort(std::vector<T>* arr, int num_threads, const Fn& cmp) {
  size_t grainsize = std::max(arr->size() / num_threads + 5, (size_t)1024*16);
  ParallelSort_(arr->data(), arr->size(), grainsize, cmp);
}

}  // namespace difacto
#endif  // DIFACTO_PARALLEL_SORT_H_
