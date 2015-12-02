#ifndef DIFACTO_TRACKER_H_
#define DIFACTO_TRACKER_H_
#include <vector>
#include <list>
#include <queue>
#include <functional>
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
    {
      std::unique_lock<std::mutex> lk(mu_);
      cond_.wait(lk, [this] {
          return pending_.size() + running_.size() == 0;
        });
    }
    done_ = true;
    cond_.notify_one();
    thread_.join();
  }

  /**
   * \brief add a list of workloads to the tracker
   * \param works
   */
  void Add(const std::vector<Work>& work) {
    CHECK(consumer_) << "set consumer first";
    mu_.lock();
    for (const auto& w : work) pending_.push(w);
    mu_.unlock();
  }

  /**
   * \brief return the number of unfinished job
   */
  int NumRemains() {
    mu_.lock();
    return pending_.size() + running_.size();
    mu_.unlock();
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
    consumer_ = consumer;
  }

 private:
  void RunConsumer() {
    while (true) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cond_.wait(lk, [this] {return (done_ || pending_.size() > 0); });
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
    mu_.lock();
    for (auto it = running_.begin(); it != running_.end(); ++it) {
      if (it->first == id) { running_.erase(it); break; }
    }
    mu_.unlock();
  }

  bool done_;
  int cur_id_;
  std::mutex mu_;
  std::condition_variable cond_;
  std::thread thread_;
  Consumer consumer_;

  std::queue<Work> pending_;
  std::list<std::pair<int, Work>> running_;
};



}  // namespace difacto
#endif  // DIFACTO_TRACKER_H_
