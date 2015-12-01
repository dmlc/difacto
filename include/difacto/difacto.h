#pragma once
#include <stdlib.h>
#include <chrono>
#include <string>
#include <functional>
#include <memory>
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "./job.h"
#include "./base.h"
#include "./model.h"
#include "./model_sync.h"
namespace difacto {

class DiFacto {
 public:
  DiFacto() { }
  ~DiFacto() { }

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
};

}  // namespace difacto
