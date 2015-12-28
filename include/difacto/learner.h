/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LEARER_H_
#define DIFACTO_LEARER_H_
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <string>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "dmlc/parameter.h"
#include "./job.h"
#include "./base.h"
#include "./store.h"
#include "./learner.h"
#include "./loss.h"
#include "./progress.h"
namespace difacto {

/**
 * \brief basic parameters for learner
 */
struct LearnerParam : public dmlc::Parameter<LearnerParam> {
  /**
   * \brief type of task,
   * - train: the training task, which the default
   * - predict: the prediction task
   * - dist_train: distributed training
   */
  std::string task;
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /**
   * \brief The optional validation dataset for a training task, either a
   *  filename or a directory
   */
  std::string val_data;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the model output for a training task */
  std::string model_out;
  /**
   * \brief the model input
   * should be specified if it is a prediction task, or a training
   */
  std::string model_in;
  /**
   * \brief the filename for prediction output.
   *  should be specified for a prediction task.
   */
  std::string pred_out;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes */
  int max_num_epochs;

  DMLC_DECLARE_PARAMETER(LearnerParam) {
    DMLC_DECLARE_FIELD(task).set_default("train");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(val_data).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(pred_out).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
  }
};

/**
 * \brief the base class of a learner
 *
 * a learner runs the learning algorithm, such as minibatch sgd
 */
class Learner {
 public:
  /** \brief construct */
  Learner();
  /** \brief deconstruct */
  virtual ~Learner();
  /**
   * \brief init learner
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  KWArgs Init(const KWArgs& kwargs);
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
  void Stop() { }
  /**
   * \brief return the current progress, thread-safe
   */
  void GetProgress(Progress* progress) {
    pmonitor_->Get(progress);;
  }

  /**
   * \brief returns the current epoch
   */
  int GetEpoch() const { return epoch_; }
  /**
   * \brief returns the current job type
   */
  int GetJobType() const { return job_type_; }

  /**
   * \brief the factory function
   * \param type the loss type such as "fm"
   */
  static Learner* Create(const std::string& type);

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
   * \param job the job received from the scheduler
   */
  virtual void Process(const Job& job) = 0;

  /** \brief paramters */
  LearnerParam param_;
  /** \brief the job tracker */
  JobTracker* tracker_;
  /** \brief callbacks for every second*/
  std::vector<Callback> cont_callbacks_;
  /** \brief callbacks for every epoch*/
  std::vector<Callback> epoch_callbacks_;
  /** \brief callbacks for before every epoch*/
  std::vector<Callback> before_epoch_callbacks_;
  /** \brief the model store*/
  Store* store_;
  /** \brief the loss*/
  Loss* loss_;
  /** \brief progress monitor collects progress */
  ProgressMonitor* pmonitor_;
  /** \brief the current epoch */
  int epoch_;
  /** \brief the current job type */
  int job_type_;
};

}  // namespace difacto
#endif  // DIFACTO_LEARNER_H_
