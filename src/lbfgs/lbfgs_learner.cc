#include "./lbfgs_learner.h"
#include "./lbfgs_utils.h"
#include "dmlc/memory_io.h"
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
  std::shared_ptr<Updater> updater(new LBFGSUpdater());
  remain = updater->Init(remain);
  // init model store
  model_store_ = Store::Create();
  model_store_->set_updater(updater);
  remain = model_store_->Init(remain);
  // init data stores
  data_store_ = new DataStore();
  remain = model_store_->Init(remain);
  tile_store_ = new TileStore(data_store_);
  // init loss
  loss_ = Loss::Create(param_.loss, nthreads_);
  remain = loss_->Init(remain);
  return remain;
}

void LBFGSLearner::RunScheduler() {
  using lbfgs::Job;
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
  IssueJobAndWait(NodeID::kWorkerGroup, Job::kInitWorker, {}, &objv);
  LL << objv;

  // iterate over data
  real_t alpha = 0;
  int epoch = param_.load_epoch >= 0 ? param_.load_epoch : 0;
  for (; epoch < param_.max_num_epochs; ++epoch) {
    // calc direction
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kPushGradient);
    std::vector<real_t> aux;
    IssueJobAndWait(NodeID::kServerGroup, Job::kPrepareCalcDirection, {alpha}, &aux);
    real_t gp;  // <∇f(w), p>
    IssueJobAndWait(NodeID::kServerGroup, Job::kCalcDirection, aux, &gp);

    // line search
    alpha = param_.alpha;
    LOG(INFO) << "epoch " << epoch << ": objv " << objv << ", <g,p> " << gp;

    std::vector<real_t> status; // = {f(w+αp), <∇f(w+αp), p>}
    for (int i = 0; i < 10; ++i) {
      status.clear();
      IssueJobAndWait(NodeID::kWorkerGroup, Job::kLinearSearch, {alpha}, &status);
      // check wolf condition
      LOG(INFO) << " - linesearch: alpha " << alpha
                << ", new_objv " << status[0] << ", <g_new,p> " << status[1];
      if ((status[0] <= objv + param_.c1 * alpha * gp) &&
          (status[1] >= param_.c2 * gp)) {
        // LOG(INFO) << "wolf condition is satisfied!";
        break;
      }
      LOG(INFO) << "wolf condition is no satisfied, decrease alpha by " << param_.rho;
      alpha *= param_.rho;
    }

    // evaluate
    objv = status[0];
    std::vector<real_t> prog;
    IssueJobAndWait(NodeID::kWorkerGroup, Job::kEvaluate, {}, &prog);

    // check stop critea
    // TODO
  }
}

void LBFGSLearner::Process(const std::string& args, std::string* rets) {
  using lbfgs::Job;
  Job job_args; job_args.ParseFromString(args);
  std::vector<real_t> job_rets;
  int type = job_args.type;
  if (type == Job::kPrepareData) {
    job_rets.push_back(PrepareData());
  } else if (type == Job::kInitServer) {
    job_rets.push_back(GetUpdater()->InitWeights());
  } else if (type == Job::kInitWorker) {
    job_rets.push_back(InitWorker());
  } else if (type == Job::kPushGradient) {
    directions_.clear();
    int t = CHECK_NOTNULL(model_store_)->Push(
        feaids_, Store::kGradient, grads_, model_offsets_);
    model_store_->Wait(t);
  } else if (type == Job::kPrepareCalcDirection) {
    GetUpdater()->PrepareCalcDirection(job_args.value[0], &job_rets);
  } else if (type == Job::kCalcDirection) {
    job_rets.push_back(GetUpdater()->CalcDirection(job_args.value));
  } else if (type == Job::kLinearSearch) {
    LinearSearch(job_args.value[0], &job_rets);
  }
  dmlc::Stream* ss = new dmlc::MemoryStringStream(rets);
  ss->Write(job_rets);
  delete ss;
}

void LBFGSLearner::IssueJobAndWait(
    int node_group, int job_type, const std::vector<real_t>& job_args,
    std::vector<real_t>* job_rets) {
  // set monitor
  Tracker::Monitor monitor = nullptr;
  if (job_rets != nullptr) {
    monitor = [job_rets](int node_id, const std::string& rets) {
      auto copy = rets; dmlc::Stream* ss = new dmlc::MemoryStringStream(&copy);
      std::vector<real_t> vec; ss->Read(&vec); delete ss;
      if (job_rets->empty()) {
        *job_rets = vec;
      } else {
        CHECK_EQ(job_rets->size(), vec.size());
        for (size_t i = 0; i < vec.size(); ++i) (*job_rets)[i] += vec[i];
      }
    };
  }
  tracker_->SetMonitor(monitor);

  // serialize and sent job
  std::pair<int, std::string> job;
  job.first = node_group;
  lbfgs::Job lbfgs_job;
  lbfgs_job.type = job_type;
  lbfgs_job.value = job_args;
  lbfgs_job.SerializeToString(&job.second);
  tracker_->Issue({job});

  // wait until finished
  while (tracker_->NumRemains() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}


size_t LBFGSLearner::PrepareData() {
  // read train data
  Reader train(param_.data_in, param_.data_format,
               model_store_->Rank(), model_store_->NumWorkers(),
               param_.data_chunk_size);
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
               param_.data_chunk_size);
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
  return nrows;
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
      feaids_, Store::kWeight, &weights_, &model_offsets_);
  model_store_->Wait(t);

  return CalcGrad(weights_, model_offsets_, &grads_);
}

void LBFGSLearner::LinearSearch(real_t alpha, std::vector<real_t>* status) {
  // w += αp
  if (directions_.empty()) {
    SArray<int> dir_offsets;
    int t = CHECK_NOTNULL(model_store_)->Pull(
        feaids_, Store::kWeight, &directions_, &model_offsets_);
    model_store_->Wait(t);
    lbfgs::Add(alpha, directions_, &weights_);
  } else {
    lbfgs::Add(alpha - alpha_, directions_, &weights_);
  }
  alpha_ = alpha;
  status->resize(2);
  (*status)[0] = CalcGrad(weights_, model_offsets_, &grads_);
  (*status)[1] = lbfgs::Inner(grads_, directions_, nthreads_);
}

real_t LBFGSLearner::CalcGrad(const SArray<real_t>& w,
                              const SArray<int>& w_offset,
                              SArray<real_t>* grad) {
  for (int i = 0; i < ntrain_blks_; ++i) {
    tile_store_->Prefetch(i, 0);
  }
  grad->resize(w.size());
  real_t objv = 0;
  for (int i = 0; i < ntrain_blks_; ++i) {
    Tile tile; tile_store_->Fetch(i, 0, &tile);
    auto data = tile.data.GetBlock();
    auto pos = GetPos(w_offset, tile.colmap);
    memset(pred_[i].data(), 0, pred_[i].size()*sizeof(real_t));
    memset(grad->data(), 0, grad->size()*sizeof(real_t));
    loss_->Predict(data, {SArray<char>(w), SArray<char>(pos)}, &pred_[i]);
    loss_->CalcGrad(
        data, {SArray<char>(pred_[i]), SArray<char>(pos), SArray<char>(w)}, grad);
    objv += loss_->CalcObjv(data.label, pred_[i]);
  }
  return objv;
}

SArray<int> LBFGSLearner::GetPos(const SArray<int>& offset, const SArray<int>& colmap) {
  if (offset.empty()) return colmap;
  SArray<int> pos(colmap.size());
  for (size_t i = 0; i < pos.size(); ++i) {
    pos[i] = offset[colmap[i]];
  }
  return pos;
}


// void LBFGSLearner::Evaluate(std::vector<real_t>* prog) {
// }

}  // namespace difacto
