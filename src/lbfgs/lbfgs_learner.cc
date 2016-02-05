#include "./lbfgs_learner.h"
#include "reader/reader.h"
namespace difacto {

KWArgs LBFGSLearner::Init(const KWArgs& kwargs) override {
}

void LBFGSLearner::RunScheduler() {
  // load data
  LOG(INFO) << "loading data... ";

  real_t data_size;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kPrepareData, {}, &data_size);
  LOG(INFO) << "loaded " << data_size << " examples";

  LOG(INFO) << "initing model... ";
  real_t model_size;
  IssueJobAndWait(NodeID::kServerGroup, Job::kInitServer, {}, &model_size);
  LOG(INFO) << "inited model with " << model_size << " parameters";

  real_t objv;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kInitWorker, {}, &fw);

  // iterate over data
  int epoch = param_.load_epoch >= 0 : param_.load_epoch + 1 : 0;
  for (; epoch < param_.max_num_epochs; ++epoch) {
    // calc direction
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kPushGradient);
    std::vector<real_t> aux;
    IssueJobAndWait(NodeID::kServerGroup, Job::kPrepareCalcDirection, {}, &aux);
    real_t gp;  // <∇f(w), p>
    IssueJobAndWait(NodeID::kServerGroup, Job::kCalcDirection, aux, &gp);

    // line search
    real_t alpha = param_.alpha;
    for (int i = 0; i < 10; ++i) {
      std::vector<real_t> status; // = {f(w+αp), <∇f(w+αp), p>}
      IssueJobAndWait(NodeID::kServerGroup + NodeID::kWorkerGroup,
                      Job::kLinearSearch, {alpha}, &status);
      // check wolf condition
      if ((stats[0] <= objv + param_.c1 * alpha * gp) &&
          (stats[1] >= param_.c2 * gp)) {
        break;
      }
      alpha *= param_.rho;
    }

    // evaluate
    objv = status[0];
    std::vector<real_t> prog;
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kEvaluate, {}, &prog);

    LL << "epoch " << epoch << ",  objv: " << objv << ", auc: " << prog[0];

    // chekc stop critea
  }
}

void LBFGSLearner::Process(const std::string& args, std::string* rets) {

  // push gradient and wait
  int t = model_store_->Push(feaids_, Store::kGradient, grads_, model_offsets_);
  model_store_->Wait(t);
}

size_t LBFGSLearner::PrepareData() {
  // read train data
  Reader train(param_.data_in, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               param_.data_chunk_size);
  tile_builder_ = new TileBuilder(tile_store_, nthreads_);
  SArray<real_t> feacnts;
  while (train.Next()) {
    auto rowblk = train.Value();
    tile_builder_->Add(rowblk, &feaids_, &feacnts);
    pred_.push_back(SArray<real_t>(rowblk.size));
    ++ntrain_blks_;
  }
  // push the feature ids and feature counts to the servers
  int t = model_store_->Push(
      feaids_, Store::kFeaCount, feacnts, SArray<int>());

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

real_t LBFGSLearner::InitWorker() {
  // remove tail features
  int filter = 0;  // TODO
  if (filter > 0) {
    SArray<real_t> feacnt;
    int t = model_store_->Pull(
        feaids_, Store::kFeaCount, &feacnt, nullptr);
    model_store_->Wait(t);

    SArray<feaid_t> filtered;
    CHECK_EQ(feacnt.size(), feaids_.size());
    for (size_t i = 0; i < feaids_.size(); ++i) {
      if (feacnt[i] > filter) filtered.push_back(feaids_[i]);
    }
    feaids_ = filtered;
  }

  // build the colmap
  CHECK_NOTNULL(tile_builder_)->BuildColmap(feaids_);

  // pull w
  int t = CHECK_NOTNULL(model_store_)->Pull(
      feaids_, Store::kWeight, &weights_, &model_offsets_);
  model_store_->Wait(t);

  return CalcGrad(weights_, model_offsets_, &grads_);
}

void LBFGSLearner::LinearSearch(real_t alpha, std::vector<real_t>* status) {
  // w += αp
  if (directions_.empty()) {
    SArray<int> dir_offsets;
    int t = CHECK_NOTNULL(model_store_)->Pull(
        feaids_, Store::kWeight, &directions_, &dir_offsets_);
    model_store_->Wait(t);
    Add(directions_, dir_offsets, alpha, model_offsets_, &weights_);
    model_offsets_ = dir_offsets;
  } else {
    Add(directions_, model_offsets_, alpha - alpha_, model_offsets_, &weights_);
  }
  alpha_ = alpha;

  status->resize(2);
  (*status)[0] = CalcGrad(weights_, model_offsets_, &grads_);

  real_t gp = 0;

#pragma omp parallel for reduction(+:gp) num_threads(nthreads_)
  for (size_t i = 0; i < grads_.size(); ++i) {
    gp += grads_[i] * directions_[i];
  }
  (*status)[1] = gp;
}

void LBFGSLearner::Evaluate(std::vector<real_t>* prog) {


}


real_t LBFGSLearner::CalcGrad(const SArray<real_t>& w,
                              const SArray<int>& w_offset,
                              SArray<real_t>* grad) {
  directions_.clear();
  for (int i = 0; i < ntrain_blks_; ++i) {
    tile_store_->Prefetch(i, 0);
  }
  grad->resize(w.size());
  real_t objv = 0;
  for (int i = 0; i < ntrain_blks_; ++i) {
    Tile tile; tile_store_->Fetch(i, 0, &tile);
    auto data = tile.data.GetBlock();
    auto pos = GetPos(w_offset, tile.colmap);
    loss_->Predict(data, {SArray<char>(w), SArray<char>(pos)}, &pred_[i]);
    loss_->CalcGrad(
        data, {SArray<char>(pred_[i]), SArray<char>(pos), SArray<char>(w)}, grad);
    objv += loss_->CalcObjv(data, pred_[i]);
  }
}

SArray<int> LBFGSLearner::GetPos(const SArray<int>& offset, const SArray<int>& colmap) {
  if (offset.empty()) return colmap;
  SArray<int> pos(colmap.size());
  for (size_t i = 0; i < pos.size(); ++i) {
    pos[i] = offset[colmap[i]];
  }
  return pos;
}

}  // namespace difacto
