/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LEANRER_H_
#define DIFACTO_LEANRER_H_
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
 * a learner runs the learning algorithm to train a model
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
   * \brief train
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
};

}  // namespace difacto
#endif  // DIFACTO_LEARNER_H_
