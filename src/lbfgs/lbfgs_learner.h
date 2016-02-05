#ifndef DIFACTO_LBFGS_LBFGS_LEARNER_H_
#define DIFACTO_LBFGS_LBFGS_LEARNER_H_
#include <string>
#include "difacto/learner.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
#include "difacto/loss.h"
#include "difacto/store.h"
#include "data/tile_store.h"
#include "data/tile_builder.h"
#include "./lbfgs_param.h"
namespace difacto {

class LBFGSLearner : public Learner {
 public:
  virtual ~LBFGSLearner() { }
  KWArgs Init(const KWArgs& kwargs) override;

 protected:
  void RunScheduler() override;
  void Process(const std::string& args, std::string* rets) override;

 private:
  /**
   * \brief send jobs to nodes and wait them finished.
   */
  void IssueJobAndWait(int node_group, const lbfgs::Job& job,
                       Tracker::Monitor monitor = nullptr);

  void IssueJobAndWait(int node_group,
                       int job_type,
                       const std::vector<real_t>& job_args = {},
                       std::vector<real_t>* job_rets = nullptr);


  /**
   * \brief preprocessing the data
   *
   * if load_epoch is set, check data cache first. otherwise, pass data once and
   * store them in an internal tile format, push feature IDs with their
   * appearance counts into the model store
   *
   * @return number examples loaded
   */
  size_t PrepareData();
  /**
   * \brief init worker
   *
   * remove features with appearance less than a threshold (if > 0).  pull w
   * from servers, calculate ∇f(w)
   *
   * @return f(w)
   */
  real_t InitWorker();

  /**
   * \brief init server
   *
   * load w if load_epoch is set. otherwise, initialize w
   *
   * @return number of model parameters
   */
  size_t InitServer() { }

  void PrepareCalcDirection(std::vector<real_t>* aux) {
  }

  real_t CalcDirection(const std::vector<real_t>& aux) {
    return 0;
  }

  void LinearSearch(real_t alpha, std::vector<real_t>* status);

  BCDUpdater* GetUpdater() {
    return CHECK_NOTNULL(std::static_pointer_cast<LBFGSUpdater>(
        CHECK_NOTNULL(model_store_)->updater()).get());
  }

  LBFGSLearnerParam param_;

  int nthreads_ = DEFAULT_NTHREADS;
  SArray<feaid_t> feaids_;
  SArray<real_t> weights_, grads_, directions_;
  SArray<int> model_offsets_;

  // data
  int ntrain_blks_ = 0;
  int nval_blks_ = 0;
  /** \brief data store */
  DataStore* data_store_ = nullptr;
  TileStore* tile_store_ = nullptr;
  TileBuilder* tile_builder_ = nullptr;

  /** \brief the model store*/
  Store* model_store_ = nullptr;

  /** \brief the loss function */
  Loss* loss_ = nullptr;
  std::vector<SArray<real_t>> pred_;

  real_t alpha_;
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_LEARNER_H_
