#include "./sgd_learner.h"
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <utility>
#include "dmlc/data.h"
#include "reader/batch_reader.h"
#include "reader/reader.h"
#include "tracker/async_local_tracker.h"
#include "data/shared_row_block_container.h"
#include "data/row_block.h"
#include "data/localizer.h"
#include "dmlc/timer.h"
#include "difacto/node_id.h"
#include "loss/bin_class_metric.h"
#include "./sgd_updater.h"

namespace difacto {

/** \brief struct to hold info for a batch job */
struct BatchJob {
  int type;
  SArray<feaid_t> feaids;
  SharedRowBlockContainer<unsigned> data;
};

void SGDLearner::RunScheduler() {
  int k = 0;
  for (; k < param_.max_num_epochs; ++k) {
    sgd::Progress train_prog;
    LOG(INFO) << "Epoch " << k << ": Training";
    RunEpoch(k, sgd::Job::kTraining, &train_prog);
    LOG(INFO) << "Done. " << train_prog.TextString();

    sgd::Progress val_prog;
    if (param_.data_val.size()) {
      LOG(INFO) << "Epoch " << k << ": Validation";
      RunEpoch(k, sgd::Job::kValidation, &val_prog);
      LOG(INFO) << "Done. " << val_prog.TextString();
    }

    // TODO stop criteria
  }
}

void SGDLearner::RunEpoch(int epoch, int job_type, sgd::Progress* prog) {
  // progress merger
  tracker_->SetMonitor(
      [this, prog](int node_id, const std::string& rets) {
        prog->Merge(rets);
      });

  // issue jobs
  int n = store_->NumWorkers() * param_.num_jobs_per_epoch;
  std::vector<std::pair<int, std::string>> jobs(n);
  for (int i = 0; i < n; ++i) {
    jobs[i].first = NodeID::kWorkerGroup;
    sgd::Job job;
    job.type = job_type;
    job.epoch = epoch;
    job.num_parts = n;
    job.part_idx = i;
    job.SerializeToString(&jobs[i].second);
  }
  tracker_->Issue(jobs);

  // wait
  while (tracker_->NumRemains()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void SGDLearner::GetPos(const SArray<int>& len,
                        SArray<int>* w_pos, SArray<int>* V_pos) {
  size_t n = len.size();
  w_pos->resize(n);
  V_pos->resize(n);
  int* w = w_pos->data();
  int* V = V_pos->data();
  int p = 0;
  for (size_t i = 0; i < n; ++i) {
    int l = len[i];
    w[i] = l == 0 ? -1 : p;
    V[i] = l > 1 ? -1 : p+1;
    p += l;
  }
}

void SGDLearner::IterateData(const sgd::Job& job, sgd::Progress* progress) {
  AsyncLocalTracker<BatchJob> batch_tracker;
  batch_tracker.SetExecutor(
      [this, progress](const BatchJob& batch,
                       const std::function<void()>& on_complete,
                       std::string* rets) {
        // use potiners here in order to copy into the callback
        auto values = new SArray<real_t>();
        auto lengths = new SArray<int>();
        auto pull_callback = [this, batch, values, lengths, progress, on_complete]() {
          // eval the objective,
          auto data = batch.data.GetBlock();
          SArray<real_t> pred(data.size);
          SArray<int> w_pos, V_pos;
          GetPos(*lengths, &w_pos, &V_pos);
          std::vector<SArray<char>> inputs = {
            SArray<char>(*values), SArray<char>(w_pos), SArray<char>(V_pos)};
          CHECK_NOTNULL(loss_)->Predict(data, inputs, &pred);
          progress->objv += loss_->Evaluate(batch.data.label.data(), pred);
          progress->nrows += pred.size();
          BinClassMetric metric(batch.data.label.data(), pred.data(),
                                pred.size(), blk_nthreads_);
          LL << metric.AUC();
          progress->auc += metric.AUC();

          // calculate the gradients
          if (batch.type == sgd::Job::kTraining) {
            SArray<real_t> grads(values->size());
            inputs.push_back(SArray<char>(pred));
            loss_->CalcGrad(data, inputs, &grads);

            // push the gradient, this task is done only if the push is complete
            store_->Push(batch.feaids,
                         Store::kGradient,
                         grads,
                         *lengths,
                         [on_complete]() { on_complete(); });
          } else {
            // a validation job
            on_complete();
          }
          delete values;
          delete lengths;
        };
        // pull the weight back
        store_->Pull(batch.feaids, Store::kWeight, values, lengths, pull_callback);
      });

  Reader* reader = nullptr;
  if (job.type == sgd::Job::kTraining) {
    reader = new BatchReader(param_.data_in,
                             param_.data_format,
                             job.part_idx,
                             job.num_parts,
                             param_.batch_size,
                             param_.batch_size * param_.shuffle,
                             param_.neg_sampling);
  } else {
    reader = new Reader(param_.data_in,
                        param_.data_format,
                        job.part_idx,
                        job.num_parts,
                        256*1024*1024);
  }
  while (reader->Next()) {
    // map feature id into continous index
    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaids = std::make_shared<std::vector<feaid_t>>();
    auto feacnt = std::make_shared<std::vector<real_t>>();
    bool push_cnt =
        job.type == sgd::Job::kTraining && job.epoch == 0;
    Localizer lc(-1, blk_nthreads_);
    lc.Compact(reader->Value(), data, feaids.get(), push_cnt ? feacnt.get() : nullptr);

    // save results into batch
    BatchJob batch;
    batch.type = job.type;
    batch.feaids = SArray<feaid_t>(feaids);
    batch.data = SharedRowBlockContainer<unsigned>(&data);
    delete data;

    // push feature count into the servers
    if (push_cnt) {
      store_->Wait(store_->Push(
          batch.feaids, Store::kFeaCount, SArray<real_t>(feacnt), {}));
    }

    // avoid too many batches are processing in parallel
    while (batch_tracker.NumRemains() > 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    batch_tracker.Issue({batch});
  }
  batch_tracker.Wait();
  delete reader;
}

KWArgs SGDLearner::Init(const KWArgs& kwargs) {
  auto remain = Learner::Init(kwargs);
  // init param
  remain = param_.InitAllowUnknown(remain);
  // init updater
  auto updater = new SGDUpdater();
  remain = updater->Init(remain);
  remain.push_back(std::make_pair("V_dim", std::to_string(updater->param().V_dim)));
  // init store
  store_ = Store::Create();
  store_->SetUpdater(std::shared_ptr<Updater>(updater));
  remain = store_->Init(remain);
  // init loss
  loss_ = Loss::Create(param_.loss, blk_nthreads_);
  remain = loss_->Init(remain);

  return remain;
}



}  // namespace difacto
