/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LEARER_H_
#define DIFACTO_LEARER_H_
#include <string.h>
#include <string>
#include <functional>
#include <vector>
#include "dmlc/io.h"
#include "./base.h"
#include "./tracker.h"
namespace difacto {

/**
 * \brief the base class of a learner
 *
 * a learner runs the learning algorithm, such as minibatch sgd
 */
class Learner {
 public:
  /**
   * \brief the factory function
   * \param type the learner type such as "sgd"
   */
  static Learner* Create(const std::string& type);
  /** \brief construct */
  Learner() { }
  /** \brief deconstruct */
  virtual ~Learner() { }
  /**
   * \brief init learner
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs);
  /**
   * \brief the callback function type
   */
  typedef std::function<void()> Callback;
  /**
   * \brief add a callback which will be evoked before running an epoch
   * @param callback the callback
   */
  void AddBeforeEpochCallback(const Callback& callback) {
    before_epoch_callbacks_.push_back(callback);
  }
  /**
   * \brief add a callback which will be evoked after an epoch is finished
   * @param callback the callback
   */
  void AddEpochCallback(const Callback& callback) {
    epoch_callbacks_.push_back(callback);
  }
  /**
   * \brief add a callback which will be evoked for every second
   * @param callback the callback
   */
  void AddContCallback(const Callback& callback) {
    cont_callbacks_.push_back(callback);
  }
  /**
   * \brief Run learner
   */
  void Run() {
    if (!IsDistributed() || !strcmp(getenv("DMLC_ROLE"), "scheduler")) {
      RunScheduler();
    } else {
      tracker_->Wait();
    }
  }
  /**
   * \brief Stop learner. It is often used to stop the training earlier
   */
  void Stop() {
    tracker_->Stop();
  }

  /**
   * \brief returns the current epoch
   */
  inline int epoch() const { return epoch_; }
  /**
   * \brief returns the current job type
   */
  inline int job_type() const { return job_type_; }


 protected:
  /**
   * \brief the function runs on the scheduler, which issues jobs to workers and
   * servers
   */
  virtual void RunScheduler() = 0;

  /**
   * \brief the function runs on the worker/server to process jobs issued by the
   * scheduler
   *
   * \param args the job arguments received from the scheduler
   * \param rets the results send back to the scheduler
   */
  virtual void Process(const std::string& args, std::string* rets) = 0;

  /** \brief the job tracker */
  Tracker* tracker_;
  /** \brief callbacks for every second*/
  std::vector<Callback> cont_callbacks_;
  /** \brief callbacks for every epoch*/
  std::vector<Callback> epoch_callbacks_;
  /** \brief callbacks for before every epoch*/
  std::vector<Callback> before_epoch_callbacks_;
  /** \brief the current epoch */
  int epoch_;
  /** \brief the current job type */
  int job_type_;
};

}  // namespace difacto
#endif  // DIFACTO_LEARNER_H_
