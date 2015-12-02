#include "difacto/difacto.h"
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include "data/batch_iter.h"
#include "data/row_block.h"
#include "common/localizer.h"
#include "dmlc/timer.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(DiFactoParam);

DiFacto::DiFacto() {
  inited_= false;
  model_ = nullptr;
  model_sync_ = nullptr;
}

DiFacto::~DiFacto() {
  delete model_;
  delete model_sync_;
}

KWArgs DiFacto::Init(const KWArgs& kwargs) {
  auto remain = param_.InitAllowUnknown(kwargs);
  local_ = param_.task.find("dist_") == std::string::npos;

  // init job tracker
  tracker_ = JobTracker::Create(local_ ? "local" : "dist");
  remain = tracker_->Init(remain);
  using namespace std::placeholders;
  tracker_->SetConsumer(std::bind(&DiFacto::Process, this, _1));

  // init model
  char* role_c = getenv("DMLC_ROLE");
  role_ = std::string(role_c, strlen(role_c));
  if (local_ || role_ == "server") {
    model_ = Model::Create(param_.algo);
    remain = model_->Init(remain);
  }

  // init model_sync
  if (local_ || role_ == "worker") {
    model_sync_ = ModelSync::Create(local_ ? "local" : "dist");
    remain = model_sync_->Init(remain);
  }

  // init loss
  loss_ = Loss::Create(param_.loss);
  remain = loss_->Init(remain);

  if (local_ && !remain.empty()) {
    LOG(WARNING) << "unrecognized keyword argument:";
    for (auto kw : remain) LOG(WARNING) << "  " << kw.first << " : " << kw.second;
  }
  return remain;
}

void DiFacto::RunScheduler() {

  // if (!local_) {
  //   printf("Connected %d servers and %d workers\n",
  //          ps::NodeInfo::NumServers(), ps::NodeInfo::NumWorkers());
  // }

  int cur_epoch = 0;

  // load model
  if (param_.model_in.size()) {
    Job job;
    job.type = Job::kLoadModel;
    job.filename = param_.model_in;
    tracker_->Add({job});
    while (tracker_->NumRemains() != 0) Sleep();
  }

  // predict
  if (param_.task.find("predict") != std::string::npos) {
    CHECK(param_.model_in.size());
    RunEpoch(cur_epoch, Job::kPrediction);
  }

  // train
  for (; cur_epoch < param_.max_num_epochs; ++ cur_epoch) {
    RunEpoch(cur_epoch, Job::kTraining);
    RunEpoch(cur_epoch, Job::kValidation);
    for (auto& cb : epoch_callbacks_) cb();
  }
}

void DiFacto::RunEpoch(int epoch, int job_type) {
  Job job;
  job.type = job_type;
  job.epoch = epoch;
  job.filename = job_type == Job::kValidation ? param_.val_data : param_.data_in;
  if (job.filename.empty()) return;

  job.num_parts = 100;
  std::vector<Job> jobs;
  for (int i = 0; i < job.num_parts; ++i) {
    job.part_idx = i;
    jobs.push_back(job);
  }
  tracker_->Add(jobs);

  while (tracker_->NumRemains() != 0) {
    Sleep();
    for (auto& cb : cont_callbacks_) cb();
  }
}

void DiFacto::Process(const Job& job) {
  if (job.type == Job::kSaveModel) {
    dmlc::Stream* fo;
    CHECK_NOTNULL(model_)->Save(fo);
  } else if (job.type == Job::kLoadModel) {
    dmlc::Stream* fi;
    CHECK_NOTNULL(model_)->Load(fi);
  } else {
    ProcessFile(job);
  }
}

void DiFacto::ProcessFile(const Job& job) {
  int batch_size = 100;
  int shuffle = 0;
  float neg_sampling = 1;
  BatchIter reader(
      job.filename, job.part_idx, job.num_parts, param_.data_format,
      batch_size, shuffle, neg_sampling);
  while (reader.Next()) {
    // map feature id into continous index
    auto batch = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaids = std::make_shared<std::vector<feaid_t>>();
    auto feacnt = std::make_shared<std::vector<real_t>>();

    bool push_cnt =
        job.type == Job::kTraining && job.epoch == 0;

    Localizer lc(param_.num_threads);
    lc.Compact(reader.Value(), batch, feaids.get(),
               push_cnt ? feacnt.get() : nullptr);

    if (push_cnt) {
      auto empty = std::make_shared<std::vector<int>>();
      model_sync_->Wait(model_sync_->Push(feaids, feacnt, empty));
    }

    // TODO
    // WaitBatch(10);
    // ProcessBatch(job.type, batch->GetBlock(), feaids, nullptr, on_complete);
  }
  // WaitBatch(1);
}


void DiFacto::ProcessBatch(
    int job_type, const dmlc::RowBlock<unsigned>& batch,
    const std::shared_ptr<std::vector<feaid_t> >& feaids,
    dmlc::Stream* pred_out, Callback on_complete) {
  auto val = new std::vector<real_t>();
  auto val_siz = new std::vector<int>();

  auto pull_callback = [this, batch, feaids, val, val_siz, on_complete,
    pred_out, job_type]() {
    double start = dmlc::GetTime();
    // eval the objective,
    CHECK_NOTNULL(loss_)->InitData(batch, *val, *val_siz);
    std::vector<real_t> prog;
    loss_->Evaluate(&prog);

    // merge into prog

    if (job_type == Job::kTraining) {
      // calculate the gradients
      loss_->CalcGrad(val);

      // push the gradient, let the system delete val, val_siz. this task is
      // done only if the push is complete
      model_sync_->Push(feaids,
                        std::shared_ptr<std::vector<real_t>>(val),
                        std::shared_ptr<std::vector<int>>(val_siz),
                        [on_complete]() { on_complete(); });
    } else {
      // save the prediction results
      if (job_type == Job::kPrediction) {
        std::vector<real_t> pred;
        loss_->Predict(&pred);
        // for (real_t p : pred) CHECK_NOTNULL(pred_out) << p << "\n";
      }

      on_complete();
      delete val;
      delete val_siz;
    }
    worktime_ += dmlc::GetTime() - start;
  };
  model_sync_->Pull(feaids, val, val_siz, pull_callback);
}

}  // namespace difacto
