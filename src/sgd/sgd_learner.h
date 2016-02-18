/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_LEARNER_H_
#define DIFACTO_SGD_SGD_LEARNER_H_
#include <string>
#include "difacto/learner.h"
#include "./sgd_utils.h"
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

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets) {
    sgd::Job job; job.ParseFromString(args);
    if (job.type == sgd::Job::kTraining ||
        job.type == sgd::Job::kValidation) {
      sgd::Progress prog;
      IterateData(job, &prog);
      prog.SerializeToString(rets);
    }
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

 private:

  void Evaluate(const SArray<dmlc::real_t>& label,
                const SArray<real_t>& pred,
                sgd::Progress* prog) {
    // TODO
  }

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
};

}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_LEARNER_H_
