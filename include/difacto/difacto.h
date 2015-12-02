#pragma once
#include <stdlib.h>
#include <chrono>
#include <string>
#include <functional>
#include <memory>
#include <thread>
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "dmlc/parameter.h"
#include "./job.h"
#include "./base.h"
#include "./model.h"
#include "./model_sync.h"
#include "./loss.h"
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
  /** \brief type of learning algorithm, default is sgd */
  std::string algo;
  /** \brief type of loss, defaut is fm*/
  std::string loss;

  /** \brief the maximal number of data passes */
  int max_num_epochs;

  /** \brief  */
  int num_threads;

  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  /** \brief  */
  DMLC_DECLARE_PARAMETER(DiFactoParam) {
    DMLC_DECLARE_FIELD(task).set_default("train");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(val_data);
    DMLC_DECLARE_FIELD(model_out);
    DMLC_DECLARE_FIELD(model_in);
    DMLC_DECLARE_FIELD(pred_out);
    DMLC_DECLARE_FIELD(algo).set_default("sgd");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
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

  void AddEpochCallback(const Callback& callback);

  void AddContCallback(const Callback& callback);

  /**
   * \brief Run difacto
   */
  void Run() {
    CHECK(inited_) << "run Init first";
    if (local_ || role_ == "scheduler") {
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
  std::vector<real_t> GetProgress() const {
    CHECK(inited_) << "run Init first";
  }

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
   * \brief process a data batch
   *
   * @param type job type
   * @param batch data batch, whose feature has been mapped into continous index
   * @param feaids the according global feature ids
   * @param pred_out data stream for saving prediction results
   * @param on_complete call the callback when this workload is finished
   */
  void ProcessBatch(int job_type,
                    const dmlc::RowBlock<unsigned>& batch,
                    const std::shared_ptr<std::vector<feaid_t> >& feaids,
                    dmlc::Stream* pred_out,
                    Callback on_complete);

  /** \brief sleep for a moment */
  inline void Sleep() { std::this_thread::sleep_for(std::chrono::seconds(1)); }

  /** \brief paramters */
  DiFactoParam param_;
  /** \brief whether or not inited */
  bool inited_;
  /** \brief the job tracker */
  JobTracker* tracker_;
  /** \brief whether of not on a single machine */
  bool local_;
  /** \brief the role (scheduler, worker, or sever) of the current process */
  std::string role_;

  /** \brief callbacks for every second*/
  std::vector<Callback> cont_callbacks_;
  /** \brief callbacks for every epoch*/
  std::vector<Callback> epoch_callbacks_;

  /** \brief the model, availabe if in the local model or this is a server */
  Model* model_;

  /** \brief the model communicator, availabe if in the local model or this is a worker */
  ModelSync* model_sync_;

  Loss* loss_;
  std::vector<real_t> progress_;

  double worktime_;
};

}  // namespace difacto
