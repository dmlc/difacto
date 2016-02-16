#ifndef DIFACTO_COMMON_THREAD_POOL_H_
#define DIFACTO_COMMON_THREAD_POOL_H_
#include <list>
#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
namespace difacto {
/**
 * \brief a pool with multiple threads
 */
class ThreadPool {
 public:
  /**
   * \brief create a threadpool
   *
   * @param num_workers number of threads
   * @param max_capacity the maximal jobs can be added to the pool
   */
  ThreadPool(int num_workers, int max_capacity = 1000000) {
    CHECK_GT(max_capacity, 0);
    CHECK_GT(num_workers, 0);
    CHECK_LT(num_workers, 100);
    capacity_ = max_capacity;
    for (int i = 0; i < num_workers; ++i) {
      workers_.push_back(std::thread(&ThreadPool::RunWorker, this));
    }
  }

  /**
   * \brief will wait all jobs are done before deconstruction
   */
  ~ThreadPool() {
    Wait();
    done_ = true;
    add_cond_.notify_all();
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i].join();
    }
  }

  /**
   * \brief add a job to the pool
   * return immmediatly if the current number of unfinished jobs is less than the
   * max_capacity. otherwise wait until the pool is available
   * @param job
   */
  void Add(const std::function<void()>& job) {
    std::unique_lock<std::mutex> lk(mu_);
    fin_cond_.wait(lk, [this]{ return tasks_.size() < capacity_; });
    tasks_.push_back(job);
    add_cond_.notify_one();
  }

  /**
   * \brief wait untill all jobs are finished
   */
  void Wait() {
    std::unique_lock<std::mutex> lk(mu_);
    fin_cond_.wait(lk, [this]{ return tasks_.empty(); });
  }

 private:
  void RunWorker() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      add_cond_.wait(lk, [this]{ return done_ || !tasks_.empty(); });
      if (done_) break;
      // run a job
      auto task = std::move(tasks_.front());
      tasks_.pop_front();
      lk.unlock();
      CHECK(task); task();
      fin_cond_.notify_all();
      lk.lock();
    }
  }
  std::atomic<bool> done_{false};
  size_t capacity_;
  std::mutex mu_;
  std::condition_variable fin_cond_, add_cond_;
  std::vector<std::thread> workers_;
  std::list<std::function<void()>> tasks_;

};
}  // namespace difacto
#endif  // DIFACTO_COMMON_THREAD_POOL_H_
