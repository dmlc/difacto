/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_TRACKER_ASYNC_LOCAL_RACKER_H_
#define DIFACTO_TRACKER_ASYNC_LOCAL_RACKER_H_
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
namespace difacto {
/**
 * \brief a thread-safe asynchronous workload tracker
 *
 * The difference to \ref Tracker:
 * 1. Only runs within a process
 * 2. Both arugment and return are templates rather than std::string
 * 3. no node_id is required
 * 4. The consumer can be asynchronous, it calls on_complete when actually finished.
 *
 * \tparam JobArgs the type of the job arguments
 * \parram JobRets the type of the job returns
 */
template<typename JobArgs, typename JobRets = std::string>
class AsyncLocalTracker {
 public:
  AsyncLocalTracker() : thread_(&AsyncLocalTracker::RunConsumer, this) { }
  ~AsyncLocalTracker() {
    Wait();
    done_ = true;
    run_cond_.notify_one();
    thread_.join();
  }

  /**
   * \brief add a list of jobs into the job queue
   * \param jobs
   */
  void Issue(const std::vector<JobArgs>& jobs) {
    CHECK(consumer_) << "set consumer first";
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& w : jobs) pending_.push(w);
    }
    run_cond_.notify_all();
  }

  /**
   * \brief block untill the number of unfinished jobers below a threadhold
   *
   * \param num_remains the maximal number of unfinished jobers
   */
  void Wait(int num_remains = 0) {
    std::unique_lock<std::mutex> lk(mu_);
    fin_cond_.wait(lk, [this, num_remains] {
        return pending_.size() + running_.size() <= num_remains;
      });
  }
  /**
   * \brief clear all jobs that have not been assigned yet
   */
  void Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    while (pending_.size()) pending_.pop();
  }
  /**
   * \brief return the number of unfinished job
   */
  int NumRemains() {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_.size() + running_.size();
  }

  /** \brief the callback function type */
  typedef std::function<void()> Callback;

  /**
   * \brief the asynchronous consumer function
   *
   * @param args the job argumetns
   * @param on_complete call it when the job is actually finished
   * @param rets  returns of the job
   */
  typedef std::function<void(
      const JobArgs args, const Callback& on_complete, JobRets* rets)> Consumer;

  /**
   * \brief set the async consumer function
   */
  void SetConsumer(const Consumer& consumer) {
    consumer_ = consumer;
  }

  /**
   * \brief the monitor
   * @param rets returns of the job
   */
  typedef std::function<void(const JobRets& rets)> Monitor;

  void SetMonitor(const Monitor& monitor) {
    monitor_ = monitor;
  }

 private:
  void RunConsumer() {
    while (true) {
      // get a job from the queue
      std::unique_lock<std::mutex> lk(mu_);
      run_cond_.wait(lk, [this] { return (done_ || pending_.size() > 0); });
      if (done_) break;
      auto it = running_.insert(std::make_pair(
          cur_id_++, std::make_pair(std::move(pending_.front()), JobRets())));
      pending_.pop();
      lk.unlock();

      // run the job
      CHECK(consumer_);
      int id = it.first->first;
      auto on_complete = [this, id]() { Remove(id); };
      consumer_(it.first->second.first, on_complete, &(it.first->second.second));
    }
  }

  inline void Remove(int id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = running_.find(id);
      CHECK(it != running_.end());
      if (monitor_) monitor_(it->second.second);
      running_.erase(it);
    }
    fin_cond_.notify_one();
  }

  bool done_ = false;
  int cur_id_ = 0;
  std::mutex mu_;
  std::condition_variable run_cond_, fin_cond_;
  std::thread thread_;
  Consumer consumer_;
  Monitor monitor_;
  std::queue<JobArgs> pending_;
  std::unordered_map<int, std::pair<JobArgs, JobRets>> running_;
};



}  // namespace difacto
#endif  // DIFACTO_TRACKER_ASYNC_LOCAL_RACKER_H_
