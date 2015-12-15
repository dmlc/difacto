/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_DIFACTO_H_
#define DIFACTO_DIFACTO_H_
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
 * \brief parameters for difacto
 */
struct DiFactoParam : public dmlc::Parameter<DiFactoParam> {
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
  /** \brief  */
  int num_threads;

  DMLC_DECLARE_PARAMETER(DiFactoParam) {
    DMLC_DECLARE_FIELD(task).set_default("train");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(val_data).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(pred_out).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
    DMLC_DECLARE_FIELD(num_threads).set_default(2);
  }
};

class DiFacto {
 public:
  /** \brief construct */
  DiFacto();
  /** \brief deconstruct */
  ~DiFacto();
  /**
   * \brief init difacto
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  KWArgs Init(const KWArgs& kwargs);

  /** \brief the callback function type */
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
   * \brief Run difacto
   */
  void Run() {
    CHECK(inited_) << "run Init first";
    if (local_ || !strcmp(getenv("DMLC_ROLE"), "scheduler")) {
      RunScheduler();
    } else {
      tracker_->Wait();
    }
  }

  /**
   * \brief Stop difacto. It is often used to stop the training earlier
   */
  void Stop() {
    CHECK(inited_) << "run Init first";
  }

  /**
   * \brief return the current progress
   */
  const Progress& progress() const { return progress_; }

  /** \brief returns the current epoch */
  int epoch() const { return epoch_; }

  /** \brief returns the current job type */
  int job_type() const { return job_type_; }

 private:
  /**
   * \brief schedule the jobs
   */
  void RunScheduler();
  /**
   * \brief schedule the jobs for one epoch
   */
  void RunEpoch(int epoch, int job_type);
  /**
   * \brief process a job
   */
  void Process(const Job& job);
  /**
   * \brief process a training/prediction job
   */
  void ProcessFile(const Job& job);

  /**
   * \brief sleep for a moment
   * \param ms the milliseconds (1e-3 sec) for sleeping
   */
  inline void Sleep(int ms = 1000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  /** \brief paramters */
  DiFactoParam param_;
  /** \brief whether or not inited */
  bool inited_;
  /** \brief the job tracker */
  JobTracker* tracker_;
  /** \brief whether of not on a single machine */
  bool local_;

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
  /** \brief current progress*/
  Progress progress_;
  ProgressPrinter pprinter_;
  double worktime_;

  int epoch_;
  int job_type_;
  // dmlc::Stream pred_out_;
};

}  // namespace difacto
#endif  // DIFACTO_DIFACTO_H_
