#pragma once

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
  Tracker() { }
  ~Tracker() { }

  /**
   * \brief add a list of workloads to the tracker
   * \param works
   */
  void Add(const std::vector<Work>& work) {

  }

  /**
   * \brief return the number of unfinished job
   */
  int NumRemains() { }

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
  Consumer consumer_;
};

}  // namespace difacto
