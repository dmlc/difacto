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
#include "common/localizer.h"
#include "dmlc/timer.h"
#include "tracker/async_local_tracker.h"
#include "difacto/loss.h"
#include "difacto/store.h"
#include "./sgd_param.h"
namespace difacto {

/**
 * \brief a sgd job
 */
struct SGDJob {
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kTraining = 3;
  static const int kValidation = 4;
  static const int kPrediction = 5;
  int type;
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;
};

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

    // // init callbacks
    // AddBeforeEpochCallback([this](){
    //     bool is_train = GetJobType() == SGDJob::kTraining;
    //     LOG(INFO) << " -- epoch " << GetEpoch() << ": "
    //               << (is_train ? "training" : "validation")
    //               << " -- ";
    //     LOG(INFO) << pprinter_.Head();
    //   });
    // AddContCallback([this]() {
    //     if (GetJobType() == SGDJob::kTraining) {
    //       Progress prog;
    //       GetProgress(&prog);
    //       LOG(INFO) << pprinter_.Body(prog);
    //     }
    //   });
    // AddEpochCallback([this](){
    //     if (GetJobType() == SGDJob::kValidation) {
    //       Progress prog;
    //       GetProgress(&prog);
    //       LOG(INFO) << pprinter_.Body(prog);
    //     }
    //   });
    return remain;
  }

 protected:
  void RunScheduler() override {
    // init progress monitor
    // pmonitor_ = ProgressMonitor::Create();

    epoch_ = 0;
    // load learner
    if (param_.model_in.size()) {
      SGDJob job;
      job.type = SGDJob::kLoadModel;
      job.filename = param_.model_in;
      // tracker_->Issue({job});
      while (tracker_->NumRemains() != 0) Sleep();
    }

    // predict
    if (param_.task.find("predict") != std::string::npos) {
      CHECK(param_.model_in.size());
      RunEpoch(epoch_, SGDJob::kPrediction);
    }

    // train
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {
      RunEpoch(epoch_, SGDJob::kTraining);
      RunEpoch(epoch_, SGDJob::kValidation);
    }
  }

  void Process(const std::string& args, std::string* rets) {
    // if (job.type == SGDJob::kSaveModel) {
    //   dmlc::Stream* fo;
    //   // CHECK_NOTNULL(learner_)->Save(fo);
    // } else if (job.type == SGDJob::kLoadModel) {
    //   dmlc::Stream* fi;
    //   // CHECK_NOTNULL(learner_)->Load(fi);
    // } else {
    //   ProcessFile(job);
    // }
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
  //   job_type_ = job_type;
  //   for (auto& cb : before_epoch_callbacks_) cb();
  //   Job job;
  //   job.type = job_type;
  //   job.epoch = epoch;
  //   job.filename = job_type == Job::kValidation ? param_.val_data : param_.data_in;
  //   if (job.filename.empty()) return;

  //   job.num_parts = 1;
  //   std::vector<Job> jobs;
  //   for (int i = 0; i < job.num_parts; ++i) {
  //     job.part_idx = i;
  //     jobs.push_back(job);
  //   }
  //   job_tracker_->Add(jobs);

  //   while (job_tracker_->NumRemains() != 0) {
  //     Sleep();
  //     for (auto& cb : cont_callbacks_) cb();
  //   }
  //   for (auto& cb : epoch_callbacks_) cb();
  }


  struct BatchJob {
    int type;
    dmlc::data::RowBlockContainer<unsigned>* data;
    std::shared_ptr<std::vector<feaid_t>> feaids;
  };

  void ProcessFile(const SGDJob& job) {
    // Tracker<BatchJob> tracker;
    // tracker.SetConsumer([this](const BatchJob& batch, const Callback& on_complete) {
    //     auto val = new std::vector<real_t>();
    //     auto val_siz = new std::vector<int>();

    //     auto pull_callback = [this, batch, val, val_siz, on_complete]() {
    //       // eval the objective,
    //       CHECK_NOTNULL(loss_)->InitData(batch.data->GetBlock(), *val, *val_siz);
    //       Progress prog; loss_->Evaluate(&prog);
    //       if (pmonitor_ == nullptr) {
    //         pmonitor_ = ProgressMonitor::Create();
    //       }
    //       pmonitor_->Add(prog);

    //       if (batch.type == Job::kTraining) {
    //         // calculate the gradients
    //         loss_->CalcGrad(val);

    //         // push the gradient, let the system delete val, val_siz. this task is
    //         // done only if the push is complete
    //         store_->Push(Store::kGradient,
    //                      batch.feaids,
    //                      std::shared_ptr<std::vector<real_t>>(val),
    //                      std::shared_ptr<std::vector<int>>(val_siz),
    //                      [on_complete]() { on_complete(); });
    //       } else {
    //         // save the prediction results
    //         if (batch.type == Job::kPrediction) {
    //           std::vector<real_t> pred;
    //           loss_->Predict(&pred);
    //           // for (real_t p : pred) CHECK_NOTNULL(pred_out) << p << "\n";
    //         }

    //         on_complete();
    //         delete val;
    //         delete val_siz;
    //       }
    //       delete batch.data;
    //     };
    //     store_->Pull(Store::kWeight, batch.feaids, val, val_siz, pull_callback);
    //   });

    // int batch_size = 100;
    // int shuffle = 0;
    // float neg_sampling = 1;
    // BatchIter reader(
    //     job.filename, param_.data_format, job.part_idx, job.num_parts,
    //     batch_size, shuffle, neg_sampling);
    // while (reader.Next()) {
    //   // map feature id into continous index
    //   BatchJob batch;
    //   batch.type = job.type;
    //   batch.data = new dmlc::data::RowBlockContainer<unsigned>();
    //   batch.feaids = std::make_shared<std::vector<feaid_t>>();
    //   auto feacnt = std::make_shared<std::vector<real_t>>();

    //   bool push_cnt =
    //       job.type == Job::kTraining && job.epoch == 0;

    //   Localizer lc(-1, 2);
    //   lc.Compact(reader.Value(), batch.data, batch.feaids.get(),
    //              push_cnt ? feacnt.get() : nullptr);

    //   if (push_cnt) {
    //     auto empty = std::make_shared<std::vector<int>>();
    //     store_->Wait(store_->Push(Store::kFeaCount, batch.feaids, feacnt, empty));
    //   }

    //   while (tracker.NumRemains() > 1) Sleep(10);

    //   tracker.Add({batch});
    // }

    // tracker.Wait();
    // // while (tracker.NumRemains() > 0) Sleep(10);
  }

 private:
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
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_SGD_LEARNER_H_
