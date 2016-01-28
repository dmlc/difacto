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
#include "loss/bin_class_metric.h"
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

  KWArgs Init(const KWArgs& kwargs) override {
    auto remain = Learner::Init(kwargs);

    // init param
    remain = param_.InitAllowUnknown(kwargs);

    // init updater
    std::shared_ptr<Updater> updater(new BCDUpdater());
    remain = updater->Init(remain);

    // init model store
    model_store_ = Store::Create();
    model_store_->set_updater(updater);
    remain = model_store_->Init(remain);

    // init data stores
    data_store_ = new DataStore();
    remain = model_store_->Init(remain);
    tile_store_ = new bcd::TileStore(data_store_);

    // init loss
    loss_ = Loss::Create("logit_delta", DEFAULT_NTHREADS);
    return remain;
  }

 protected:

  void RunScheduler() override {
    // load data
    std::vector<real_t> feagrp_avg;
    bcd::JobArgs load; load.type = bcd::JobArgs::kPrepareData;
    auto load_monitor = [&feagrp_avg](int node_id, const std::string& rets) {
      bcd::PrepDataRets prets; prets.ParseFromString(rets);
      bcd::Add(prets.feagrp_avg, &feagrp_avg);
    };
    IssueJobAndWait(NodeID::kWorkerGroup, load, load_monitor);

    // partition feature group and build feature map
    bcd::JobArgs build; build.type = bcd::JobArgs::kBuildFeatureMap;
    int nworker = model_store_->NumWorkers();
    std::vector<std::pair<int,int>> feagrp;
    for (int i = 0; i < static_cast<int>(feagrp_avg.size()); ++i) {
      int nblk = static_cast<int>(std::ceil(feagrp_avg[i] / nworker));
      if (nblk > 0) feagrp.push_back(std::make_pair(i, nblk));
    }
    bcd::FeatureBlock::Partition(
        param_.num_feature_group_bits, feagrp, &build.fea_blk_ranges);
    IssueJobAndWait(NodeID::kWorkerGroup, build);

    // iterate over data
    std::vector<int> feablks(build.fea_blk_ranges.size());
    for (size_t i = 0; i < feablks.size(); ++i) feablks[i] = i;
    epoch_ = 0;
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {

      bcd::JobArgs iter; iter.type = bcd::JobArgs::kIterateData;
      std::random_shuffle(feablks.begin(), feablks.end());
      iter.fea_blks = feablks;
      bcd::Progress progress;
      auto iter_monitor = [&progress](int node_id, const std::string& rets) {
        bcd::Progress prog; prog.ParseFromString(rets);
        progress.Add(node_id, prog);
      };
      IssueJobAndWait(NodeID::kWorkerGroup | NodeID::kServerGroup, iter, iter_monitor);

      LL << "objv: " << progress.value[0] << ", auc: " << progress.value[1];
    }
  }

  void Process(const std::string& args, std::string* rets) {
    bcd::JobArgs job(args);
    if (job.type == bcd::JobArgs::kPrepareData) {
      LL << "prepare";
      bcd::PrepDataRets prets;
      PrepareData(job, &prets);
      prets.SerializeToString(rets);
    } else if (job.type == bcd::JobArgs::kBuildFeatureMap) {
      LL << "build";
      BuildFeatureMap(job);
      LL << "done";
    } else if (job.type == bcd::JobArgs::kIterateData) {
      bcd::Progress prog;
      IterateData(job, &prog);
      prog.SerializeToString(rets);
    }

    // TODO save and load
  }

 private:
  /**
   * \brief sleep for a moment
   * \param ms the milliseconds (1e-3 sec) for sleeping
   */
  inline void Sleep(int ms = 1000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  /**
   * \brief send jobs to nodes and wait them finished.
   */
  void IssueJobAndWait(int node_group, const bcd::JobArgs& job,
                       Tracker::Monitor monitor = nullptr) {
    // TODO
    int nworker = model_store_->NumWorkers();
    std::vector<std::pair<int, std::string>> jobs(nworker);
    for (int i = 0; i < nworker; ++i) {
      jobs[i].first = NodeID::Encode(NodeID::kWorkerGroup, i);
      job.SerializeToString(&(jobs[i].second));
    }
    tracker_->SetMonitor(monitor);
    tracker_->Issue(jobs);
    while (tracker_->NumRemains() != 0) { Sleep(); }
  }

  void PrepareData(const bcd::JobArgs& job,
                   bcd::PrepDataRets* rets) {
    // read train data
    int chunk_size = 1<<28;  // read and process a 512MB chunk each time
    ChunkIter train(param_.data_in, param_.data_format,
                    model_store_->Rank(), model_store_->NumWorkers(),
                    chunk_size);
    bcd::FeaGroupStats stats(param_.num_feature_group_bits);
    tile_builder_ = new bcd::TileBuilder(tile_store_);
    while (train.Next()) {
      auto rowblk = train.Value();
      stats.Add(rowblk);
      tile_builder_->Add(rowblk, true);
      pred_.push_back(SArray<real_t>(rowblk.size));
      ++ntrain_blks_;
    }
    // push the feature ids and feature counts to the servers
    int t = model_store_->Push(
        tile_builder_->feaids, Store::kFeaCount, tile_builder_->feacnts, SArray<int>());
    // report statistics to the scheduler
    stats.Get(&(CHECK_NOTNULL(rets)->feagrp_avg));

    // read validation data if any
    if (param_.data_val.size()) {
      ChunkIter val(param_.data_val, param_.data_format,
                    model_store_->Rank(), model_store_->NumWorkers(),
                    chunk_size);

      while (val.Next()) {
        auto rowblk = val.Value();
        tile_builder_->Add(rowblk, false);
        pred_.push_back(SArray<real_t>(rowblk.size));
        ++nval_blks_;
      }
    }

    // wait the previous push finished
    model_store_->Wait(t);
    tile_builder_->feacnts.clear();
  }

  void BuildFeatureMap(const bcd::JobArgs& job) {
    CHECK_NOTNULL(tile_builder_);
    // pull the aggregated feature counts from the servers
    SArray<real_t> feacnt;
    int t = model_store_->Pull(
        tile_builder_->feaids, Store::kFeaCount, &feacnt, nullptr);
    model_store_->Wait(t);

    // remove the filtered features
    SArray<feaid_t> filtered;
    size_t n = tile_builder_->feaids.size();
    CHECK_EQ(feacnt.size(), n);
    for (size_t i = 0; i < n; ++i) {
      if (feacnt[i] > param_.tail_feature_filter) {
        filtered.push_back(tile_builder_->feaids[i]);
      }
    }

    // build colmap for each rowblk
    tile_builder_->feaids = filtered;
    tile_builder_->BuildColmap(job.fea_blk_ranges);
    delete tile_builder_; tile_builder_ = nullptr;

    // init aux data
    std::vector<Range> pos;
    bcd::FeatureBlock::FindPosition(filtered, job.fea_blk_ranges, &pos);
    feaids_.resize(pos.size());
    delta_.resize(pos.size());
    model_offset_.resize(pos.size());

    for (size_t i = 0; i < pos.size(); ++i) {
      feaids_[i] = filtered.segment(pos[i].begin, pos[i].end);
      bcd::Delta::Init(feaids_[i].size(), &delta_[i]);
    }
  }

  void IterateData(const bcd::JobArgs& job, bcd::Progress* progress) {
    CHECK(job.fea_blks.size());
    // hint for data prefetch
    for (int f : job.fea_blks) {
      for (int d = 0; d < ntrain_blks_ + nval_blks_; ++d) {
        tile_store_->Prefetch(d, f);
      }
    }

    size_t nfeablk = job.fea_blks.size();
    int tau = 0;
    bcd::BlockTracker feablk_tracker(nfeablk);
    for (size_t i = 0; i < nfeablk; ++i) {
      auto on_complete = [&feablk_tracker, i]() {
        feablk_tracker.Finish(i);
      };
      int f = job.fea_blks[i];
      IterateFeablk(f, on_complete, progress);

      if (i >= tau) feablk_tracker.Wait(i - tau);
    }
    for (int i = nfeablk - tau; i < nfeablk ; ++i) feablk_tracker.Wait(i);
  }

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
                     bcd::Progress* progress) {
    // 1. calculate gradient
    SArray<real_t> grad;
    for (int i = 0; i < ntrain_blks_; ++i) {
      CalcGrad(i, blk_id, model_offset_[blk_id], &grad);
    }

    // 3. once push is done, pull the changes for the weights
    // this callback will be called when the push is finished
    auto push_callback = [this, blk_id, progress, on_complete]() {
      // must use pointer here, since it may be reallocated by model_store_
      SArray<real_t>* delta_w = new SArray<real_t>();
      SArray<int>* delta_w_offset = new SArray<int>();
      // 4. once the pull is done, update the prediction
      // the callback will be called when the pull is finished
      auto pull_callback = [this, blk_id, delta_w, delta_w_offset, progress, on_complete]() {
        model_offset_[blk_id] = *delta_w_offset;
        for (int i = 0; i < ntrain_blks_ + nval_blks_; ++i) {
          UpdtPred(i, blk_id, *delta_w_offset, *delta_w, progress);
        }
        delete delta_w;
        delete delta_w_offset;
        on_complete();
      };
      // pull the changes of w from the servers
      model_store_->Pull(
          feaids_[blk_id], Store::kWeight, delta_w, delta_w_offset, pull_callback);
    };
    // 2. push gradient to the servers
    auto grad_offset = model_offset_[blk_id];
    for (auto& o : grad_offset) o += o;
    model_store_->Push(
        feaids_[blk_id], Store::kGradient, grad, grad_offset, push_callback);
  }

  void CalcGrad(int rowblk_id, int colblk_id,
                const SArray<int>& model_offset,
                SArray<real_t>* grad) {
    // load data
    bcd::Tile tile;
    tile_store_->Fetch(rowblk_id, colblk_id, &tile);

    // build index
    size_t n = tile.colmap.size();  // n can be 0
    bool no_os = model_offset.empty();
    SArray<int> grad_pos(n);
    SArray<real_t> delta(n);
    for (size_t i = 0; i < n; ++i) {
      int map = tile.colmap[i];
      if (map < 0) {
        grad_pos[i] = -1;
      } else {
        grad_pos[i] = no_os ? map * 2 : model_offset[map] * 2;
        delta[i] = delta_[colblk_id][map];
      }
    }
    if (delta.empty()) delta = delta_[colblk_id];
    grad->resize(no_os ? n * 2 : model_offset.back() * 2);

    // calc grad
    loss_->CalcGrad(tile.data.GetBlock(), {SArray<char>(pred_[rowblk_id]),
            SArray<char>(grad_pos), SArray<char>(delta)}, grad);
  }

  void UpdtPred(int rowblk_id, int colblk_id,
                const SArray<int> delta_w_offset,
                const SArray<real_t> delta_w,
                bcd::Progress* progress) {
    // load data
    bcd::Tile tile;
    tile_store_->Fetch(rowblk_id, colblk_id, &tile);

    // build index and update delta_
    size_t n = tile.colmap.size();
    bool no_os = delta_w_offset.empty();
    SArray<int> w_pos(n);
    for (size_t i = 0; i < n; ++i) {
      int map = tile.colmap[i];
      if (map < 0) {
        w_pos[i] = -1;
      } else {
        w_pos[i] = no_os ? map : delta_w_offset[map];
        bcd::Delta::Update(delta_w[w_pos[i]], &delta_[colblk_id][map]);
      }
    }
    if (n == 0) {
      CHECK_EQ(delta_[colblk_id].size(), delta_w.size());
      for (size_t i = 0; i < delta_w.size(); ++i) {
        bcd::Delta::Update(delta_w[i], &delta_[colblk_id][i]);
      }
    }

    // predict
    loss_->Predict(tile.data.GetBlock(),
                   {SArray<char>(delta_w_offset), SArray<char>(w_pos)},
                   &pred_[rowblk_id]);

    // evaluate
    CHECK_EQ(tile.data.label.size(), pred_[rowblk_id].size());
    BinClassMetric metric(tile.data.label.data(),
                          pred_[rowblk_id].data(),
                          pred_[rowblk_id].size());
    auto& val = progress->value;
    if (val.empty()) val.resize(3);
    val[0] = metric.LogitObjv();
    val[1] = metric.AUC();
  }

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
  std::vector<SArray<real_t>> delta_;
  std::vector<SArray<int>> model_offset_;
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_BCD_LEARNER_H_
