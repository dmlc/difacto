#ifndef DEFACTO_LEARNER_BCD_LEARNER_H_
#define DEFACTO_LEARNER_BCD_LEARNER_H_
#include <cmath>
#include <algorithm>
#include "difacto/learner.h"
#include "difacto/node_id.h"
#include "dmlc/data.h"
#include "data/chunk_iter.h"
#include "data/data_store.h"
#include "./bcd_param.h"
#include "./bcd_job.h"
#include "./bcd_utils.h"
#include "./bcd_updater.h"
#include "./tile_store.h"
#include "./tile_builder.h"
#include "loss/logit_loss_delta.h"
namespace difacto {

class BCDLearner : public Learner {
 public:
  BCDLearner() {}
  virtual ~BCDLearner() {
    delete model_store_;
    delete data_store_;
    delete tile_store_;
    delete loss_;
  }

  KWArgs Init(const KWArgs& kwargs) override;

  void AddEpochEndCallback(
      const std::function<void(int epoch, const bcd::Progress& prog)>& callback) {
    epoch_end_callback_.push_back(callback);
  }

 protected:
  void RunScheduler() override;

  void Process(const std::string& args, std::string* rets);

 private:
  /**
   * \brief sleep for a moment
   * \param ms the milliseconds (1e-3 sec) for sleeping
   */
  inline void Sleep(int ms = 5) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  /**
   * \brief send jobs to nodes and wait them finished.
   */
  void IssueJobAndWait(int node_group, const bcd::JobArgs& job,
                       Tracker::Monitor monitor = nullptr);

  void PrepareData(const bcd::JobArgs& job, bcd::PrepDataRets* rets);

  void BuildFeatureMap(const bcd::JobArgs& job);

  void IterateData(const bcd::JobArgs& job, bcd::Progress* progress);

  /**
   * \brief iterate a feature block
   *
   * the logic is as following
   *
   * 1. calculate gradident
   * 2. push gradients to servers, so servers will update the weight
   * 3. once the push is done, pull the changes for the weights back from
   *    the servers
   * 4. once the pull is done update the prediction
   *
   * however, two things make the implementation is not so intuitive.
   *
   * 1. we need to iterate the data block one by one for both calcluating
   * gradient and update prediction
   * 2. we used callbacks to avoid to be blocked by the push and pull.
   *
   * NOTE: once cannot iterate on the same block before it is actually finished.
   *
   * @param blk_id
   * @param on_complete will be called when actually finished
   */
  void IterateFeablk(int blk_id,
                     const std::function<void()>& on_complete,
                     bcd::Progress* progress);

  void CalcGrad(int rowblk_id, int colblk_id,
                const SArray<int>& grad_offset,
                SArray<real_t>* grad);

  void UpdtPred(int rowblk_id, int colblk_id,
                const SArray<int> delta_w_offset,
                const SArray<real_t> delta_w,
                bcd::Progress* progress);

  /** \brief the current epoch */
  int epoch_ = 0;
  int ntrain_blks_ = 0;
  int nval_blks_ = 0;

  /** \brief the model store*/
  Store* model_store_ = nullptr;
  Loss* loss_ = nullptr;
  /** \brief data store */
  DataStore* data_store_ = nullptr;
  bcd::TileStore* tile_store_ = nullptr;
  bcd::TileBuilder* tile_builder_ = nullptr;

  /** \brief parameters */
  BCDLearnerParam param_;

  std::vector<SArray<real_t>> pred_;
  std::vector<SArray<feaid_t>> feaids_;
  std::vector<Range> feablk_pos_;
  std::vector<SArray<real_t>> delta_;
  std::vector<SArray<int>> model_offset_;
  std::vector<std::function<void(int epoch, const bcd::Progress& prog)>> epoch_end_callback_;
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_BCD_LEARNER_H_
