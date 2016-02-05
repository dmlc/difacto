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
   * pass data once and store them in an internal tile format, push feature IDs
   * with their appearance counts into the model store
   *
   * @return number examples loaded
   */
  size_t PrepareData();
  /**
   * \brief remove the tail features
   *
   * remove features with appearance less than a threshold
   */
  void RemoveTailFeatures();

  void CalcGradient();

  void CalcDirection();

  void LinearSearch();

  LBFGSLearnerParam param_;

  struct Job {
    static const int kPrepareData = 1;
    static const int kInitServer = 2;
    static const int kInitWorker = 3;
    static const int kPushGradient = 4;
    static const int kPrepareCalcDirection = 5;
    static const int kCalcDirection = 6;
    static const int kLinearSearch = 7;
    static const int kSaveModel = 8;
  };

  int nthreads_ = DEFAULT_NTHREADS;
  SArray<feaid_t> feaids_;

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
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_LEARNER_H_
