/**
 *  Copyright (c) 2015 by Contributors
 */
#include "./lbfgs_learner.h"
#include "./lbfgs_utils.h"
#include "difacto/node_id.h"
#include "reader/reader.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(LBFGSLearnerParam);
DMLC_REGISTER_PARAMETER(LBFGSUpdaterParam);

KWArgs LBFGSLearner::Init(const KWArgs& kwargs) {
  auto remain = Learner::Init(kwargs);
  // init param
  remain = param_.InitAllowUnknown(kwargs);
  // init updater
  auto updater = new LBFGSUpdater();
  remain = updater->Init(remain);
  remain.push_back(std::make_pair("V_dim", std::to_string(updater->param().V_dim)));
  // init model store
  model_store_ = Store::Create();
  model_store_->SetUpdater(std::shared_ptr<Updater>(updater));
  remain = model_store_->Init(remain);
  // init data stores
  tile_store_ = new TileStore();
  remain = tile_store_->Init(remain);
  // init loss
  loss_ = Loss::Create(param_.loss, nthreads_);
  remain = loss_->Init(remain);
  return remain;
}

void LBFGSLearner::RunScheduler() {
  using lbfgs::Job;
  LOG(INFO) << "scaning data... ";
  std::vector<real_t> data;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kPrepareData, {}, &data);
  LOG(INFO) << "found " << data[0] << " examples, splitted into " << data[1] << " chunks";

  LOG(INFO) << "initing... ";
  std::vector<real_t> server;
  IssueJobAndWait(NodeID::kServerGroup, Job::kInitServer, {}, &server);
  LOG(INFO) << "inited model with " << server[1] << " parameters";

  std::vector<real_t> worker;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kInitWorker, {}, &worker);
  real_t objv = server[0] + worker[0];

  // iterate over data
  real_t alpha = 0;
  int k = param_.load_epoch >= 0 ? param_.load_epoch : 0;
  for (; k < param_.max_num_epochs; ++k) {
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kPushGradient);

    std::vector<real_t> B;
    IssueJobAndWait(NodeID::kServerGroup, Job::kPrepareCalcDirection, {alpha}, &B);

    std::vector<real_t> p_gf;  // = <p, ∂f(w)>
    IssueJobAndWait(NodeID::kServerGroup, Job::kCalcDirection, B, &p_gf);

    alpha = param_.alpha;
    LOG(INFO) << "epoch " << k << ": objv " << objv << ", <p,g> " << p_gf[0];

    std::vector<real_t> status;  // = {f(w+αp), <p, ∂f(w+αp)>}
    for (int i = 0; i < param_.max_num_linesearchs; ++i) {
      status.clear();
      IssueJobAndWait(NodeID::kWorkerGroup + NodeID::kServerGroup,
                      Job::kLineSearch, {alpha}, &status);
      // check wolf condition
      LOG(INFO) << " - linesearch: alpha " << alpha
                << ", new_objv " << status[0] << ", <p,g_new> " << status[1];
      if ((status[0] <= objv + param_.c1 * alpha * p_gf[0]) &&
          (status[1] >= param_.c2 * p_gf[0])) {
        break;
      }
      alpha *= param_.rho;
    }
    objv = status[0];

    // evaluate
    // std::vector<real_t> eval;
    // IssueJobAndWait(NodeID::kWorkerGroup, Job::kEvaluate, {}, &eval);

    lbfgs::Progress prog;
    prog.objv = objv;
    for (const auto& cb : epoch_end_callback_) cb(k, prog);
    // check stop critea
    // TODO(mli)
  }
}

void LBFGSLearner::Process(const std::string& args, std::string* rets) {
  using lbfgs::Job;
  Job job_args; job_args.ParseFromString(args);
  std::vector<real_t> job_rets;
  int type = job_args.type;
  if (type == Job::kPrepareData) {
    PrepareData(&job_rets);
  } else if (type == Job::kInitServer) {
    GetUpdater()->InitWeight(&job_rets);
  } else if (type == Job::kInitWorker) {
    job_rets.push_back(InitWorker());
  } else if (type == Job::kPushGradient) {
    directions_.clear();
    int t = CHECK_NOTNULL(model_store_)->Push(
        feaids_, Store::kGradient, grads_, model_lens_);
    model_store_->Wait(t);
  } else if (type == Job::kPrepareCalcDirection) {
    GetUpdater()->PrepareCalcDirection(&job_rets);
  } else if (type == Job::kCalcDirection) {
    job_rets.push_back(GetUpdater()->CalcDirection(job_args.value));
  } else if (type == Job::kLineSearch) {
    if (IsWorker()) LineSearch(job_args.value[0], &job_rets);
    if (IsServer()) GetUpdater()->LineSearch(job_args.value[0], &job_rets);
  }
  dmlc::Stream* ss = new dmlc::MemoryStringStream(rets);
  ss->Write(job_rets);
  delete ss;
}

void LBFGSLearner::PrepareData(std::vector<real_t>* rets) {
  // read train data
  size_t chunk_size = static_cast<size_t>(param_.data_chunk_size * 1024 * 1024);
  Reader train(param_.data_in, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               chunk_size);
  size_t nrows = 0;
  tile_builder_ = new TileBuilder(tile_store_, nthreads_);
  SArray<real_t> feacnts;
  while (train.Next()) {
    auto rowblk = train.Value();
    nrows += rowblk.size;
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
               chunk_size);
    while (val.Next()) {
      auto rowblk = val.Value();
      nrows += rowblk.size;
      tile_builder_->Add(rowblk);
      pred_.push_back(SArray<real_t>(rowblk.size));
      ++nval_blks_;
    }
  }

  // wait the previous push finished
  model_store_->Wait(t);
  rets->resize(2);
  (*rets)[0] = nrows;
  (*rets)[1] = ntrain_blks_ + nval_blks_;
}

real_t LBFGSLearner::InitWorker() {
  // remove tail features
  int filter = GetUpdater()->param().tail_feature_filter;
  if (filter > 0) {
    SArray<real_t> feacnt;
    int t = model_store_->Pull(
        feaids_, Store::kFeaCount, &feacnt, nullptr);
    model_store_->Wait(t);

    SArray<feaid_t> filtered;
    lbfgs::RemoveTailFeatures(feaids_, feacnt, filter, &filtered);
    feaids_ = filtered;
  }

  // build the colmap
  CHECK_NOTNULL(tile_builder_)->BuildColmap(feaids_);

  // pull w
  int t = CHECK_NOTNULL(model_store_)->Pull(
      feaids_, Store::kWeight, &weights_, &model_lens_);
  model_store_->Wait(t);

  return CalcGrad(weights_, model_lens_, &grads_);
}

void LBFGSLearner::LineSearch(real_t alpha, std::vector<real_t>* status) {
  // w += αp
  if (directions_.empty()) {
    SArray<int> dir_lens;
    int t = CHECK_NOTNULL(model_store_)->Pull(
        feaids_, Store::kWeight, &directions_, &model_lens_);
    model_store_->Wait(t);
    alpha_ = 0;
  }
  lbfgs::Add(alpha - alpha_, directions_, &weights_);
  alpha_ = alpha;
  status->resize(2);
  (*status)[0] += CalcGrad(weights_, model_lens_, &grads_);
  (*status)[1] += lbfgs::Inner(grads_, directions_, nthreads_);
}

real_t LBFGSLearner::CalcGrad(const SArray<real_t>& w_val,
                              const SArray<int>& w_len,
                              SArray<real_t>* grad) {
  for (int i = 0; i < ntrain_blks_; ++i) {
    tile_store_->Prefetch(i, 0);
  }
  grad->resize(w_val.size());
  memset(grad->data(), 0, grad->size()*sizeof(real_t));
  real_t objv = 0;
  real_t a = 0;
  for (int i = 0; i < ntrain_blks_; ++i) {
    // prepare data
    Tile tile; tile_store_->Fetch(i, 0, &tile);
    auto data = tile.data.GetBlock();
    SArray<int> w_pos, V_pos;
    GetPos(w_len, tile.colmap, &w_pos, &V_pos);
    memset(pred_[i].data(), 0, pred_[i].size()*sizeof(real_t));
    std::vector<SArray<char>> param = {
      SArray<char>(w_val), SArray<char>(w_pos), SArray<char>(V_pos)};

    // calc
    loss_->Predict(data, param, &pred_[i]);
    param.push_back(SArray<char>(pred_[i]));
    loss_->CalcGrad(data, param, grad);
    objv += loss_->Evaluate(data.label, pred_[i]);
    a += Norm2(pred_[i]);
  }
  return objv;
}

void LBFGSLearner::GetPos(const SArray<int>& len, const SArray<int>& colmap,
                          SArray<int>* w_pos, SArray<int>* V_pos) {
  size_t n = colmap.size();
  V_pos->resize(n, -1);
  if (len.empty()) { *w_pos = colmap; return; }
  w_pos->resize(n, -1);

  int* w = w_pos->data();
  int* V = V_pos->data();
  int const* e = len.data();
  int i = 0, p = 0;
  for (size_t j = 0; j < n; ++j) {
    if (colmap[j] == -1) continue;
    for (; i < colmap[j]; ++i) { p += *e; ++e; }
    w[j] = p;
    V[j] = *e > 1 ? p+1 : -1;
  }
}

}  // namespace difacto
