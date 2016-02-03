#ifndef DEFACTO_LEARNER_SGD_LEARNER_H_
#define DEFACTO_LEARNER_SGD_LEARNER_H_
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <thread>
#include "dmlc/data.h"
#include "difacto/learner.h"
#include "data/batch_iter.h"
#include "data/row_block.h"
#include "data/localizer.h"
#include "dmlc/timer.h"
#include "difacto/loss.h"
#include "difacto/store.h"
#include "difacto/node_id.h"
#include "difacto/reporter.h"
#include "./sgd_param.h"
#include "./sgd_job.h"
#include "data/shared_row_block_container.h"
#include "tracker/async_local_tracker.h"
namespace difacto {

class SGDLearner : public Learner {
 public:
  SGDLearner() {
    store_ = nullptr;
    loss_ = nullptr;
  }

  virtual ~SGDLearner() {
    delete loss_;
    delete store_;
  };

  KWArgs Init(const KWArgs& kwargs) override {
    auto remain = Learner::Init(kwargs);
    // init param
    remain = param_.InitAllowUnknown(remain);
    // init store
    store_ = Store::Create();
    remain = store_->Init(remain);
    // init loss
    loss_ = Loss::Create(param_.loss);
    remain = loss_->Init(remain);
    // init reporter
    reporter_ = Reporter::Create();
    using namespace std::placeholders;
    reporter_->SetMonitor(std::bind(&SGDLearner::ProgressMonitor, this, _1, _2));
    // init callbacks

    before_epoch_callbacks_.push_back([this](){
        bool is_train = job_type_ == sgd::Job::kTraining;
        LOG(INFO) << " -- epoch " << epoch_ << ": "
                  << (is_train ? "training" : "validation") << " -- ";
        LOG(INFO) << sgd::Progress::TextHead();
      });
    cont_callbacks_.push_back([this]() {
        if (job_type_ == sgd::Job::kTraining) LOG(INFO) << progress_.TextString();
      });
    epoch_callbacks_.push_back([this](){
        if (job_type_ == sgd::Job::kValidation) LOG(INFO) << progress_.TextString();
      });
    return remain;
  }

 protected:
  void RunScheduler() override {
    epoch_ = 0;
    // train
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {
      RunEpoch(epoch_, sgd::Job::kTraining);
      RunEpoch(epoch_, sgd::Job::kValidation);
    }
  }

  void Process(const std::string& args, std::string* rets) {
    sgd::Job job(args);
    if (job.type == sgd::Job::kTraining ||
        job.type == sgd::Job::kValidation) {
      IterateData(job);
    }
  }

 private:
  /**
   * \brief sleep for a moment
   * \param ms the milliseconds (1e-3 sec) for sleeping
   */
  inline void Sleep(int ms = 1000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  void RunEpoch(int epoch, int job_type) {
    // issue jobs
    job_type_ = job_type;
    for (auto& cb : before_epoch_callbacks_) cb();
    sgd::Job job;
    job.type = job_type;
    job.epoch = epoch;
    job.filename = job_type == sgd::Job::kValidation ? param_.val_data : param_.data_in;
    job.num_parts = store_->NumWorkers() * param_.job_size;
    if (job.filename.empty()) return;

    std::vector<std::pair<int, std::string>> jobs(job.num_parts);
    for (int i = 0; i < job.num_parts; ++i) {
      jobs[i].first = NodeID::kWorkerGroup;
      job.part_idx = i;
      job.SerializeToString(&jobs[i].second);
    }
    tracker_->Issue(jobs);

    // wait until finished
    while (tracker_->NumRemains() != 0) {
      Sleep();
      for (auto& cb : cont_callbacks_) cb();
    }
    for (auto& cb : epoch_callbacks_) cb();
  }

  /** \brief struct to hold info for a batch job */
  struct BatchJob {
    int type;
    SArray<feaid_t> feaids;
    SharedRowBlockContainer<unsigned> data;
  };
  /**
   * \brief iterate on a part of a data
   *
   * it repeats the following steps
   *
   * 1. read batch_size examples
   * 2. preprogress data (map from uint64 feature index into continous ones)
   * 3. pull the newest model for this batch from the servers
   * 4. compute the gradients on this batch
   * 5. push the gradients to the servers to update the model
   *
   * to maximize the parallelization of i/o and computation, we uses three
   * threads here. they are asynchronized by callbacks
   *
   * a. main thread does 1 and 2
   * b. batch_tracker's thread does 3 once a batch is preprocessed
   * c. store_'s threads does 4 and 5 when the weight is pulled back
   */
  void IterateData(const sgd::Job& job) {
    AsyncLocalTracker<BatchJob> batch_tracker;
    batch_tracker.SetExecutor([this](const BatchJob& batch,
                                     const std::function<void()>& on_complete,
                                     std::string* rets) {
        // use potiners here in order to copy into the callback
        auto values = new SArray<real_t>();
        auto offsets = new SArray<int>();
        auto pull_callback = [this, batch, values, offsets, on_complete]() {
          // eval the objective,
          SArray<real_t> pred;
          CHECK_NOTNULL(loss_)->Predict(
              batch.data.GetBlock(),
              {SArray<char>(*values), SArray<char>(*offsets)},
              &pred);
          Evaluate(batch.data.label, pred);

          if (batch.type == sgd::Job::kTraining) {
            // calculate the gradients, reuse values to store the gradients
            loss_->CalcGrad(
                batch.data.GetBlock(),
                {SArray<char>(pred), SArray<char>(*offsets), SArray<char>(*values)},
                values);

            // push the gradient, this task is done only if the push is complete
            store_->Push(batch.feaids,
                         Store::kGradient,
                         *values,
                         *offsets,
                         [on_complete]() { on_complete(); });
          } else {
            // a validation job
            on_complete();
          }
          delete values;
          delete offsets;
        };
        // pull the weight back
        store_->Pull(batch.feaids, Store::kWeight, values, offsets, pull_callback);
      });

    int batch_size = 100;
    int shuffle = 0;
    float neg_sampling = 1;
    BatchIter reader(
        job.filename, param_.data_format, job.part_idx, job.num_parts,
        batch_size, shuffle, neg_sampling);
    while (reader.Next()) {
      // map feature id into continous index
      auto data = new dmlc::data::RowBlockContainer<unsigned>();
      auto feaids = std::make_shared<std::vector<feaid_t>>();
      auto feacnt = std::make_shared<std::vector<real_t>>();
      bool push_cnt =
          job.type == sgd::Job::kTraining && job.epoch == 0;
      Localizer lc(-1, 2);
      lc.Compact(reader.Value(), data, feaids.get(), push_cnt ? feacnt.get() : nullptr);

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
      while (batch_tracker.NumRemains() > 1) Sleep(10);

      batch_tracker.Issue({batch});
    }
    batch_tracker.Wait();
  }

 private:
  void Evaluate(const SArray<real_t>& label, const SArray<real_t>& pred) {
    sgd::Progress prog;
    // TODO
    std::string report; prog.SerializeToString(&report);
    reporter_->Report(report);
  }

  void ProgressMonitor(int node_id, const std::string& report) {
    sgd::Progress prog; prog.ParseFromString(report);
    progress_.Merge(node_id, prog);
  }

  /** \brief the model store*/
  Store* store_;
  /** \brief the loss*/
  Loss* loss_;
  /** \brief parameters */
  SGDLearnerParam param_;
  // ProgressPrinter pprinter_;

  /** \brief the current epoch */
  int epoch_;
  /** \brief the current job type */
  int job_type_;

  Reporter* reporter_;
  sgd::Progress progress_;

  /** \brief callbacks for every second*/
  std::vector<std::function<void()>> cont_callbacks_;
  /** \brief callbacks for every epoch*/
  std::vector<std::function<void()>> epoch_callbacks_;
  /** \brief callbacks for before every epoch*/
  std::vector<std::function<void()>> before_epoch_callbacks_;
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_SGD_LEARNER_H_
