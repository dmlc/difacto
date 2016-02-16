/**
 *  Copyright (c) 2015 by Contributors
 */
#include "./lbfgs_learner.h"
#include "./lbfgs_utils.h"
#include "difacto/node_id.h"
#include "loss/bin_class_metric.h"
#include "reader/reader.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(LBFGSLearnerParam);
DMLC_REGISTER_PARAMETER(LBFGSUpdaterParam);

void LBFGSLearner::RunScheduler() {
  // init
  using lbfgs::Job;
  LOG(INFO) << "Staring training using L-BFGS with " << nthreads_ << " threads";
  LOG(INFO) << "Scaning data... ";
  std::vector<real_t> data;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kPrepareData, {}, &data);
  real_t ntrain = data[0], nval = data[3];
  LOG(INFO) << " - found " << ntrain << " training examples, splitted into "
            << data[1] << " chunks";
  if (nval > 0) {
    LOG(INFO) << " - found " << nval << " validation examples, splitted into "
              << data[4] << " chunks";
  }
  return;
  std::vector<real_t> server;
  IssueJobAndWait(NodeID::kServerGroup, Job::kInitServer, {}, &server);
  LOG(INFO) << "Inited model with " << server[1] << " parameters";
  std::vector<real_t> worker;
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kInitWorker, {}, &worker);
  real_t objv = server[0] + worker[0];

  // iterate over data
  real_t alpha = 0, val_auc = 0, new_objv = 0;
  int k = param_.load_epoch >= 0 ? param_.load_epoch : 0;
  for (; k < param_.max_num_epochs; ++k) {
    LOG(INFO) << "Epoch " << k << ":";
    // calc direction
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kPushGradient);
    std::vector<real_t> B;
    IssueJobAndWait(NodeID::kServerGroup, Job::kPrepareCalcDirection, {alpha}, &B);
    std::vector<real_t> p_gf;  // = <p, ∂f(w)>
    IssueJobAndWait(NodeID::kServerGroup, Job::kCalcDirection, B, &p_gf);
    // start linesearch
    LOG(INFO) << " - start linesearch with objv = " << objv <<
        ", <p,g> = " << p_gf[0];
    alpha = k == 0 ? (ntrain / data[2]) : param_.alpha;
    std::vector<real_t> status;  // = {f(w+αp), <p, ∂f(w+αp)>}
    for (int i = 0; i < param_.max_num_linesearchs; ++i) {
      status.clear();
      IssueJobAndWait(NodeID::kWorkerGroup + NodeID::kServerGroup,
                      Job::kLineSearch, {alpha}, &status);
      // check wolf condition
      new_objv = status[0];
      LOG(INFO) << " - alpha = " << alpha
                << ", objv = " << status[0] << ", <p,g> = " << status[1];
      if ((new_objv <= objv + param_.c1 * alpha * p_gf[0]) &&
          (status[1] >= param_.c2 * p_gf[0])) {
        LOG(INFO) << " - wolfe condition is satisifed";
        break;  // satisified
      }
      if (i+1 == param_.max_num_linesearchs) {
        LOG(INFO) << " - reach the maximal number of linesearch steps [" << i+1 << "]";
      }
      alpha *= param_.rho;
    }
    // evaluate AUC, ...
    std::vector<real_t> eval;
    IssueJobAndWait(NodeID::kWorkerGroup + NodeID::kServerGroup,
                    Job::kEvaluate, {}, &eval);
    lbfgs::Progress prog; prog.ParseFromVector(eval);
    prog.objv = new_objv;
    prog.auc /= ntrain;
    LOG(INFO) << " - training AUC = " << prog.auc;
    if (nval > 0) {
      prog.val_auc /= nval;
      LOG(INFO) << " - validation AUC = " << prog.val_auc;
    }
    for (const auto& cb : epoch_end_callback_) cb(k, prog);

    // check stop critea
    real_t eps = fabs(new_objv - objv) / objv;
    if (eps < param_.stop_rel_objv) {
      LOG(INFO) << "Change of objective [" << eps << "] < stop_rel_objv ["
                << param_.stop_rel_objv << "]";
      break;
    }
    if (nval > 0) {
      eps = prog.val_auc - val_auc;
      if (eps < param_.stop_val_auc) {
        LOG(INFO) << "Change of validation AUC [" << eps << "] < stop_val_auc ["
                  << param_.stop_val_auc << "]";
        break;
      }
    }
    if (k+1 == param_.max_num_epochs) {
      LOG(INFO) << "Reach maximal number of epochs";
    }
    objv = new_objv;
    val_auc = prog.val_auc;
  }
  LOG(INFO) << "Training is done";
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
  } else if (type == Job::kEvaluate) {
    lbfgs::Progress prog;
    if (IsWorker()) Evaluate(&prog);
    if (IsServer()) GetUpdater()->Evaluate(&prog);
    prog.SerializeToVector(&job_rets);
  } else {
    LOG(FATAL) << "unknown job type " << type;
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
  size_t nrows = 0, nnz = 0;
  tile_builder_ = new TileBuilder(tile_store_, nthreads_);
  SArray<real_t> feacnts;
  while (train.Next()) {
    auto rowblk = train.Value();
    nrows += rowblk.size;
    nnz += rowblk.offset[rowblk.size];
    tile_builder_->Add(rowblk, &feaids_, &feacnts);
    pred_.push_back(SArray<real_t>(rowblk.size));
    ++ntrain_blks_;
  }
  rets->resize(6);
  (*rets)[0] = nrows;
  (*rets)[1] = ntrain_blks_;
  (*rets)[2] = nnz;

  tile_builder_->Wait();
  // push the feature ids and feature counts to the servers
  int t = model_store_->Push(
      feaids_, Store::kFeaCount, feacnts, SArray<int>());

  // read validation data if any
  if (param_.data_val.size()) {
    nrows = 0; nnz = 0;
    Reader val(param_.data_val, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               chunk_size);
    while (val.Next()) {
      auto rowblk = val.Value();
      nrows += rowblk.size;
      nnz += rowblk.offset[rowblk.size];
      tile_builder_->Add(rowblk);
      pred_.push_back(SArray<real_t>(rowblk.size));
      ++nval_blks_;
    }
    (*rets)[3] = nrows;
    (*rets)[4] = nval_blks_;
    (*rets)[5] = nnz;
  }
  tile_builder_->Wait();
  // wait the previous push finished
  model_store_->Wait(t);
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
  prog_.auc = 0;
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

    BinClassMetric metric(data.label, pred_[i].data(), pred_[i].size(), nthreads_);
    prog_.auc += metric.AUC();
  }
  return objv;
}

void LBFGSLearner::Evaluate(lbfgs::Progress* prog) {
  *prog = prog_;
  prog->val_auc = 0;
  // validation data
  for (int i = ntrain_blks_; i < ntrain_blks_ + nval_blks_; ++i) {
    // prepare data
    Tile tile; tile_store_->Fetch(i, 0, &tile);
    auto data = tile.data.GetBlock();
    SArray<int> w_pos, V_pos;
    GetPos(model_lens_, tile.colmap, &w_pos, &V_pos);
    memset(pred_[i].data(), 0, pred_[i].size()*sizeof(real_t));
    std::vector<SArray<char>> param = {
      SArray<char>(weights_), SArray<char>(w_pos), SArray<char>(V_pos)};
    loss_->Predict(data, param, &pred_[i]);
    BinClassMetric metric(data.label, pred_[i].data(), pred_[i].size(), nthreads_);
    prog->val_auc += metric.AUC();
  }
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

KWArgs LBFGSLearner::Init(const KWArgs& kwargs) {
  auto remain = Learner::Init(kwargs);
  // init param
  remain = param_.InitAllowUnknown(kwargs);
  nthreads_ = param_.num_threads <= 0 ? std::thread::hardware_concurrency() : param_.num_threads;
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

}  // namespace difacto
