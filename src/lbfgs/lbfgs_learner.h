/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LBFGS_LBFGS_LEARNER_H_
#define DIFACTO_LBFGS_LBFGS_LEARNER_H_
#include <string>
#include <vector>
#include "difacto/learner.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
#include "difacto/loss.h"
#include "difacto/store.h"
#include "data/tile_store.h"
#include "data/tile_builder.h"
#include "common/learner_utils.h"
#include "./lbfgs_param.h"
#include "./lbfgs_utils.h"
#include "./lbfgs_updater.h"
namespace difacto {

class LBFGSLearner : public Learner {
 public:
  virtual ~LBFGSLearner() {
    delete model_store_;
    delete tile_store_;
    delete loss_;
  }
  KWArgs Init(const KWArgs& kwargs) override;

  void AddEpochEndCallback(
      const std::function<void(int epoch, const lbfgs::Progress& prog)>& callback) {
    epoch_end_callback_.push_back(callback);
  }

  LBFGSUpdater* GetUpdater() {
    return CHECK_NOTNULL(std::static_pointer_cast<LBFGSUpdater>(
        CHECK_NOTNULL(model_store_)->updater()).get());
  }

 protected:
  void RunScheduler() override;
  void Process(const std::string& args, std::string* rets) override;

 private:
  /**
   * \brief send jobs to nodes and wait them finished.
   */
  void IssueJobAndWait(int node_group,
                       int job_type,
                       const std::vector<real_t>& job_args = {},
                       std::vector<real_t>* job_rets = nullptr) {
    lbfgs::Job job; job.type = job_type; job.value = job_args;
    std::string args; job.SerializeToString(&args);
    SendJobAndWait(node_group, args, tracker_, job_rets);
  }

  /**
   * \brief preprocessing the data
   *
   * if load_epoch is set, check data cache first. otherwise, pass data once and
   * store them in an internal tile format, push feature IDs with their
   * appearance counts into the model store
   *
   */
  void PrepareData(std::vector<real_t>* rets);
  /**
   * \brief init worker
   *
   * remove features with appearance less than a threshold (if > 0).  pull w
   * from servers, calculate ∇f(w)
   *
   * @return f(w)
   */
  real_t InitWorker();

  real_t CalcGrad(const SArray<real_t>& w_val,
                  const SArray<int>& w_len,
                  SArray<real_t>* grad);

  void LineSearch(real_t alpha, std::vector<real_t>* status);

  void GetPos(const SArray<int>& len, const SArray<int>& colmap,
              SArray<int>* w_pos, SArray<int>* V_pos);
  LBFGSLearnerParam param_;
  int nthreads_ = DEFAULT_NTHREADS;
  SArray<feaid_t> feaids_;
  SArray<real_t> weights_, grads_, directions_;
  SArray<int> model_lens_;

  // data
  int ntrain_blks_ = 0;
  int nval_blks_ = 0;
  /** \brief data store */
  TileStore* tile_store_ = nullptr;
  TileBuilder* tile_builder_ = nullptr;

  /** \brief the model store*/
  Store* model_store_ = nullptr;

  /** \brief the loss function */
  Loss* loss_ = nullptr;
  std::vector<SArray<real_t>> pred_;

  real_t alpha_;

  std::vector<std::function<void(
      int epoch, const lbfgs::Progress& prog)>> epoch_end_callback_;
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_LEARNER_H_
