/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_TRACKER_TRACKER_H_
#define DIFACTO_TRACKER_TRACKER_H_
#include <vector>
#include <utility>
#include <list>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
namespace difacto {

/**
 * \brief a thread-safe workload tracker
 *
 * - the producer can continously add workloads into the tracker
 * - the tracker continous call consumer to run the workload using its' own thread
 * \tparam Workload the workload type
 */
template<typename Work>
class Tracker {
 public:
  Tracker() : done_(false),
              cur_id_(0),
              thread_(&Tracker::RunConsumer, this) { }
  ~Tracker() {
    Wait();
    done_ = true;
    run_cond_.notify_one();
    thread_.join();
  }

  /**
   * \brief add a list of workloads to the tracker
   * \param works
   */
  void Add(const std::vector<Work>& work) {
    CHECK(consumer_) << "set consumer first";
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& w : work) pending_.push(w);
    }
    run_cond_.notify_all();
  }

  /**
   * \brief block untill the number of unfinished workers below a threadhold
   *
   * \param num_remains the maximal number of unfinished workers
   */
  void Wait(int num_remains = 0) {
    std::unique_lock<std::mutex> lk(mu_);
    fin_cond_.wait(lk, [this, num_remains] {
        return pending_.size() + running_.size() <= num_remains;
      });
  }
  /**
   * \brief clear all works that have not been assigned yet
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
   * @param work  the work
   * @param on_complete call it when the work is actually finished
   */
  typedef std::function<void(
      const Work& work, const Callback& on_complete)> Consumer;

  /**
   * \brief set the async consumer function
   */
  void SetConsumer(const Consumer& consumer) {
    LL << "asdf";
    consumer_ = consumer;
  }

 private:
  void RunConsumer() {
    while (true) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        run_cond_.wait(lk, [this] { return (done_ || pending_.size() > 0); });
        if (done_) break;
        running_.push_back(
            std::make_pair(cur_id_++, std::move(pending_.front())));
        pending_.pop();
      }
      CHECK(consumer_);
      int id = running_.back().first;
      consumer_(running_.back().second, [this, id]() {
          Remove(id);
        });
    }
  }

  void Remove(int id) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto it = running_.begin(); it != running_.end(); ++it) {
        if (it->first == id) { running_.erase(it); break; }
      }
    }
    fin_cond_.notify_one();
  }

  bool done_;
  int cur_id_;
  std::mutex mu_;
  std::condition_variable run_cond_, fin_cond_;
  std::thread thread_;
  Consumer consumer_;

  std::queue<Work> pending_;
  std::list<std::pair<int, Work>> running_;
};



}  // namespace difacto
#endif  // DIFACTO_TRACKER_TRACKER_H_
