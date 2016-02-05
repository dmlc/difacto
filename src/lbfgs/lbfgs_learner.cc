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

void LBFGSLearner::RemoveTailFeatures() {
  int filter = 0;  // TODO

  if (filter > 0) {
    // remove tail features
    SArray<real_t> feacnt;
    int t = model_store_->Pull(
        feaids_, Store::kFeaCount, &feacnt, nullptr);
    model_store_->Wait(t);

    SArray<feaid_t> filtered;
    size_t n = feaids_.size();
    CHECK_EQ(feacnt.size(), n);
    for (size_t i = 0; i < n; ++i) {
      if (feacnt[i] > filter) {
        filtered.push_back(feaids_[i]);
      }
    }
    feaids_ = filtered;
  }

  // build the colmap
  CHECK_NOTNULL(tile_builder_);
  tile_builder_->BuildColmap(feaids_);
}


void LBFGSLearner::CalcGrad() {
  // pretech data
  for (int i = 0; i < ntrain_blks_; ++i) {
    tile_store_->Prefetch(i, 0);
  }

  SArray<real_t> grad(weights_.size());
  for (int i = 0; i < ntrain_blks_; ++i) {
    // load data
    Tile tile; tile_store_->Fetch(i, 0, &tile);

    // build grad_pos
    bool no_pos = model_offsets_.empty();
    SArray<int> grad_pos(no_pos ? 0 : tile.colmap.size());
    for (size_t i = 0; i < grad_pos.size(); ++i) {
      grad_pos[i] = model_offsets_[tile.colmap[i]];
    }

    // calc grad
    auto param = {SArray<char>(pred_[i]), SArray<char>(grad_pos), SArray<char>(wegihts_)};
    loss_->CalcGrad(tile.data.GetBlock(), param, *grad);
  }

  // push gradient and wait
  int t = model_store_->Push(feaids_, Store::kGradient, grad, model_offsets_);
  model_store_->Wait(t);
}

}  // namespace difacto
