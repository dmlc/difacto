/**
 *  Copyright (c) 2015 by Contributors
 */
#include "./bcd_learner.h"
#include <algorithm>
#include <cmath>
#include "difacto/node_id.h"
#include "reader/reader.h"
#include "loss/bin_class_metric.h"
#include "./bcd_updater.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(BCDUpdaterParam);

KWArgs BCDLearner::Init(const KWArgs& kwargs) {
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
  tile_store_ = new TileStore();
  remain = tile_store_->Init(remain);
  // init loss
  loss_ = Loss::Create("logit_delta", DEFAULT_NTHREADS);
  remain = loss_->Init(remain);
  return remain;
}

void BCDLearner::Process(const std::string& args, std::string* rets) {
  using bcd::Job;
  Job job_args; job_args.ParseFromString(args);
  int type = job_args.type;
  std::vector<real_t> job_rets;
  if (type == Job::kPrepareData) {
    PrepareData(&job_rets);
  } else if (type == Job::kBuildFeatureMap) {
    BuildFeatureMap(job_args.feablk_ranges);
  } else if (type == Job::kIterateData) {
    IterateData(job_args.feablks, &job_rets);
  }
  dmlc::Stream* ss = new dmlc::MemoryStringStream(rets);
  ss->Write(job_rets);
  delete ss;
}
void BCDLearner::RunScheduler() {
  using bcd::Job;
  // load data
  LOG(INFO) << "loading data... ";
  Job load; load.type = Job::kPrepareData;
  std::vector<real_t> load_rets;
  IssueJobAndWait(NodeID::kWorkerGroup, load, &load_rets);
  LOG(INFO) << "loaded " << load_rets.back() << " examples";

  // partition feature group and build feature map
  Job build; build.type = Job::kBuildFeatureMap;
  std::vector<std::pair<int, int>> feagrp;
  int nfeablk = load_rets.size()-2;
  for (int i = 0; i < nfeablk; ++i) {
    int nblk = static_cast<int>(std::ceil(
        load_rets[i] / load_rets[nfeablk] * param_.block_ratio));
    if (nblk > 0) feagrp.push_back(std::make_pair(i, nblk));
  }
  bcd::PartitionFeature(
      param_.num_feature_group_bits, feagrp, &build.feablk_ranges);
  LOG(INFO) << "partitioning feature into " << build.feablk_ranges.size() << " blocks";
  IssueJobAndWait(NodeID::kWorkerGroup, build);

  // iterate over data
  std::vector<int> feablks(build.feablk_ranges.size());
  for (size_t i = 0; i < feablks.size(); ++i) feablks[i] = i;
  epoch_ = 0;
  for (; epoch_ < param_.max_num_epochs; ++epoch_) {
    std::random_shuffle(feablks.begin(), feablks.end());
    Job iter; iter.type = Job::kIterateData;
    iter.feablks = feablks;
    std::vector<real_t> progress;
    IssueJobAndWait(NodeID::kWorkerGroup + NodeID::kServerGroup, iter, &progress);
    for (const auto& cb : epoch_end_callback_) {
      cb(epoch_, progress);
    }
    real_t cnt = progress[0];
    LL << "epoch: " << epoch_
       << ", objv: " << progress[1] / cnt
       << ", auc: " << progress[2] / cnt
       << ", acc: " << progress[3] / cnt;
  }
}


void BCDLearner::PrepareData(std::vector<real_t>* fea_stats) {
  // read train data
  Reader train(param_.data_in, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               param_.data_chunk_size);
  bcd::FeaGroupStats stats(param_.num_feature_group_bits);
  tile_builder_ = new TileBuilder(tile_store_, DEFAULT_NTHREADS, true);
  SArray<real_t> feacnts;
  while (train.Next()) {
    auto rowblk = train.Value();
    stats.Add(rowblk);
    tile_builder_->Add(rowblk, &feaids_, &feacnts);
    pred_.push_back(SArray<real_t>(rowblk.size));
    ++ntrain_blks_;
  }
  // push the feature ids and feature counts to the servers
  int t = model_store_->Push(
      feaids_, Store::kFeaCount, feacnts, SArray<int>());
  // report statistics to the scheduler
  stats.Get(fea_stats);

  // read validation data if any
  if (param_.data_val.size()) {
    Reader val(param_.data_val, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               param_.data_chunk_size);
    while (val.Next()) {
      auto rowblk = val.Value();
      tile_builder_->Add(rowblk);
      pred_.push_back(SArray<real_t>(rowblk.size));
      ++nval_blks_;
    }
  }

  // wait the previous push finished
  model_store_->Wait(t);
}

void BCDLearner::BuildFeatureMap(const std::vector<Range>& feablk_ranges) {
  CHECK_NOTNULL(tile_builder_);
  // pull the aggregated feature counts from the servers
  SArray<real_t> feacnt;
  int t = model_store_->Pull(
      feaids_, Store::kFeaCount, &feacnt, nullptr);
  model_store_->Wait(t);

  // remove the filtered features
  SArray<feaid_t> filtered;
  size_t n = feaids_.size();
  CHECK_EQ(feacnt.size(), n);
  int filter = std::static_pointer_cast<BCDUpdater>(
      model_store_->updater())->param().tail_feature_filter;
  for (size_t i = 0; i < n; ++i) {
    if (feacnt[i] > filter) {
      filtered.push_back(feaids_[i]);
    }
  }
  feaids_.clear();

  // build colmap for each rowblk
  std::vector<Range> pos;
  tile_builder_->BuildColmap(filtered, feablk_ranges, &pos);
  delete tile_builder_; tile_builder_ = nullptr;

  // init feature blocks
  feablks_.resize(pos.size());

  for (size_t i = 0; i < pos.size(); ++i) {
    auto& feablk = feablks_[i];
    feablk.feaids = filtered.segment(pos[i].begin, pos[i].end);
    bcd::Delta::Init(feablk.feaids.size(), &feablk.delta);
    feablk.pos = pos[i];
  }
}

void BCDLearner::IterateData(const std::vector<int>& feablks,
                             std::vector<real_t>* progress) {
  CHECK(feablks.size());
  // hint for data prefetch
  for (int f : feablks) {
    for (int d = 0; d < ntrain_blks_ + nval_blks_; ++d) {
      tile_store_->Prefetch(d, f);
    }
  }

  size_t nfeablk = feablks.size();
  int tau = 0;
  bcd::BlockTracker feablk_tracker(nfeablk);
  for (size_t i = 0; i < nfeablk; ++i) {
    auto on_complete = [&feablk_tracker, i]() {
      feablk_tracker.Finish(i);
    };
    int f = feablks[i];
    IterateFeablk(f, on_complete, (i == nfeablk -1 ? progress : nullptr));

    if (i >= tau) feablk_tracker.Wait(i - tau);
  }
  for (int i = nfeablk - tau; i < nfeablk ; ++i) feablk_tracker.Wait(i);
}

void BCDLearner::IterateFeablk(int blk_id,
                               const std::function<void()>& on_complete,
                               std::vector<real_t>* progress) {
  // 1. calculate gradient
  auto& feablk = feablks_[blk_id];
  SArray<int> grad_offset = feablk.model_offset;
  for (int& o : grad_offset) o += o;  // it's ok to overwrite model_offset_[blk_id]
  SArray<real_t> grad(
      grad_offset.empty() ? feablk.feaids.size() * 2 : grad_offset.back());
  for (int i = 0; i < ntrain_blks_; ++i) {
    CalcGrad(i, blk_id, grad_offset, &grad);
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
      feablks_[blk_id].model_offset = *delta_w_offset;
      for (int i = 0; i < ntrain_blks_ + nval_blks_; ++i) {
        UpdtPred(i, blk_id, *delta_w_offset, *delta_w, progress);
      }
      delete delta_w;
      delete delta_w_offset;
      on_complete();
    };
    // pull the changes of w from the servers
    model_store_->Pull(
        feablks_[blk_id].feaids, Store::kWeight, delta_w, delta_w_offset, pull_callback);
  };
  // 2. push gradient to the servers
  model_store_->Push(
      feablk.feaids, Store::kGradient, grad, grad_offset, push_callback);
}


void BCDLearner::CalcGrad(int rowblk_id, int colblk_id,
                          const SArray<int>& grad_offset,
                          SArray<real_t>* grad) {
  // load data
  Tile tile; tile_store_->Fetch(rowblk_id, colblk_id, &tile);

  // build index
  size_t n = tile.colmap.size();
  bool no_os = grad_offset.empty();
  SArray<int> grad_pos(n);
  SArray<real_t> delta(n);
  auto& feablk = feablks_[colblk_id];
  int pos_begin = feablk.pos.begin;
  for (size_t i = 0; i < n; ++i) {
    int map = tile.colmap[i];
    if (map < 0) {
      grad_pos[i] = -1;
    } else {
      map -= pos_begin; CHECK_GE(map, 0);
      grad_pos[i] = no_os ? map * 2 : grad_offset[map] * 2;
      delta[i] = feablk.delta[map];
    }
  }

  // calc grad
  loss_->CalcGrad(tile.data.GetBlock(), {SArray<char>(pred_[rowblk_id]),
          SArray<char>(grad_pos), SArray<char>(delta)}, grad);
}

void BCDLearner::UpdtPred(int rowblk_id, int colblk_id,
                          const SArray<int> delta_w_offset,
                          const SArray<real_t> delta_w,
                          std::vector<real_t>* progress) {
  // load data
  Tile tile;
  tile_store_->Fetch(rowblk_id, colblk_id, &tile);
  size_t n = tile.colmap.size();

  // build index and update delta_
  bool no_os = delta_w_offset.empty();
  SArray<int> w_pos(n);
  auto& feablk = feablks_[colblk_id];
  int pos_begin = feablk.pos.begin;
  for (size_t i = 0; i < n; ++i) {
    int map = tile.colmap[i];
    if (map < 0) {
      w_pos[i] = -1;
    } else {
      map -= pos_begin; CHECK_GE(map, 0);
      w_pos[i] = no_os ? map : delta_w_offset[map];
      bcd::Delta::Update(delta_w[w_pos[i]], &feablk.delta[map]);
    }
  }

  // predict
  loss_->Predict(tile.data.GetBlock(),
                 {SArray<char>(delta_w), SArray<char>(w_pos)},
                 &pred_[rowblk_id]);

  // evaluate
  if (!progress) return;
  CHECK_EQ(tile.data.label.size(), pred_[rowblk_id].size());
  BinClassMetric metric(tile.data.label.data(),
                        pred_[rowblk_id].data(),
                        pred_[rowblk_id].size());

  // value[0] : count
  // value[1] : objv
  // value[2] : auc
  // value[3] : acc
  // value[4] : ...
  auto& val = *progress;
  if (val.empty()) val.resize(4);
  val[0] += tile.data.label.size();
  val[1] += metric.LogitObjv();
  val[2] += metric.AUC();
  val[3] += metric.Accuracy(.5);
}

}  // namespace difacto
