/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_LEARNER_H_
#define DIFACTO_SGD_SGD_LEARNER_H_
#include <string>
#include "difacto/learner.h"
#include "./sgd_utils.h"
#include "./sgd_updater.h"
#include "./sgd_param.h"
#include "difacto/loss.h"
#include "difacto/store.h"
namespace difacto {

class SGDLearner : public Learner {
 public:
  SGDLearner() {
    store_ = nullptr;
    loss_ = nullptr;
  }

  virtual ~SGDLearner() {
    delete loss_;
    delete store_;
  }
  KWArgs Init(const KWArgs& kwargs) override;

  void AddEpochEndCallback(const std::function<void(
      int epoch, const sgd::Progress& train, const sgd::Progress& val)>& callback) {
    epoch_end_callback_.push_back(callback);
  }

  SGDUpdater* GetUpdater() {
    return CHECK_NOTNULL(std::static_pointer_cast<SGDUpdater>(
        CHECK_NOTNULL(store_)->updater()).get());
  }

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets) {
    using sgd::Job;
    sgd::Progress prog;
    Job job; job.ParseFromString(args);
    if (job.type == Job::kTraining ||
        job.type == Job::kValidation) {
      IterateData(job, &prog);
    } else if (job.type == Job::kEvaluation) {
      GetUpdater()->Evaluate(&prog);
    }
    prog.SerializeToString(rets);
  }

 private:
  void RunEpoch(int epoch, int job_type, sgd::Progress* prog);

  /**
   * \brief iterate on a part of a data
   *
   * it repeats the following steps
   *
   * 1. read batch_size examples
   * 2. preprogress data (map from uint64 feature index into continous ones)
   * 3. pull the newest model for this batch from the servers
   * 4. compute the gradients on this batch
   * 5. push the gradients to the servers to update the model
   *
   * to maximize the parallelization of i/o and computation, we uses three
   * threads here. they are asynchronized by callbacks
   *
   * a. main thread does 1 and 2
   * b. batch_tracker's thread does 3 once a batch is preprocessed
   * c. store_'s threads does 4 and 5 when the weight is pulled back
   */
  void IterateData(const sgd::Job& job, sgd::Progress* prog);

  real_t EvaluatePenalty(const SArray<real_t>& weight,
                         const SArray<int>& w_pos,
                         const SArray<int>& V_pos);
  void GetPos(const SArray<int>& len,
              SArray<int>* w_pos, SArray<int>* V_pos);
  /** \brief the model store*/
  Store* store_;
  /** \brief the loss*/
  Loss* loss_;
  /** \brief parameters */
  SGDLearnerParam param_;
  // ProgressPrinter pprinter_;
  int blk_nthreads_ = DEFAULT_NTHREADS;

  std::vector<std::function<void(int epoch, const sgd::Progress& train,
                                 const sgd::Progress& val)>> epoch_end_callback_;
};

}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_LEARNER_H_
