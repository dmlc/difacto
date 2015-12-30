#ifndef DEFACTO_LEARNER_SGD_LEARNER_H_
#define DEFACTO_LEARNER_SGD_LEARNER_H_
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <thread>
#include "dmlc/data.h"
#include "dmlc/parameter.h"
#include "difacto/learner.h"
#include "data/batch_iter.h"
#include "data/row_block.h"
#include "common/localizer.h"
#include "dmlc/timer.h"
#include "tracker/tracker.h"
#include "difacto/loss.h"
#include "difacto/store.h"
namespace difacto {

struct SGDLearnerParam : public dmlc::Parameter<SGDLearnerParam> {
  /**
   * \brief type of task,
   * - train: the training task, which is the default
   * - predict: the prediction task
   */
  std::string task;
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /**
   * \brief The optional validation dataset for a training task, either a
   *  filename or a directory
   */
  std::string val_data;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the model output for a training task */
  std::string model_out;
  /**
   * \brief the model input
   * should be specified if it is a prediction task, or a training
   */
  std::string model_in;
  /**
   * \brief the filename for prediction output.
   *  should be specified for a prediction task.
   */
  std::string pred_out;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes */
  int max_num_epochs;

  /**
   * \brief the minibatch size
   */
  int batch_size;
  int shuffle;
  float neg_sampling;

  DMLC_DECLARE_PARAMETER(SGDLearnerParam) {
    DMLC_DECLARE_FIELD(task).set_default("train");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(val_data).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(pred_out).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
  }
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

    // init callbacks
    AddBeforeEpochCallback([this](){
        LOG(INFO) << " -- epoch " << GetEpoch() << ": "
                  << (GetJobType() == Job::kTraining ? "training" : "validation")
                  << " -- ";
        LOG(INFO) << pprinter_.Head();
      });
    AddContCallback([this]() {
        if (GetJobType() == Job::kTraining) {
          Progress prog;
          GetProgress(&prog);
          LOG(INFO) << pprinter_.Body(prog);
        }
      });
    AddEpochCallback([this](){
        if (GetJobType() == Job::kValidation) {
          Progress prog;
          GetProgress(&prog);
          LOG(INFO) << pprinter_.Body(prog);
        }
      });
    return remain;
  }

 protected:
  void RunScheduler() override {
    // init progress monitor
    pmonitor_ = ProgressMonitor::Create();

    epoch_ = 0;
    // load learner
    if (param_.model_in.size()) {
      Job job;
      job.type = Job::kLoadModel;
      job.filename = param_.model_in;
      job_tracker_->Add({job});
      while (job_tracker_->NumRemains() != 0) Sleep();
    }

    // predict
    if (param_.task.find("predict") != std::string::npos) {
      CHECK(param_.model_in.size());
      RunEpoch(epoch_, Job::kPrediction);
    }

    // train
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {
      RunEpoch(epoch_, Job::kTraining);
      RunEpoch(epoch_, Job::kValidation);
    }
  }

  void Process(const Job& job) {
    if (job.type == Job::kSaveModel) {
      dmlc::Stream* fo;
      // CHECK_NOTNULL(learner_)->Save(fo);
    } else if (job.type == Job::kLoadModel) {
      dmlc::Stream* fi;
      // CHECK_NOTNULL(learner_)->Load(fi);
    } else {
      ProcessFile(job);
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
    job_type_ = job_type;
    for (auto& cb : before_epoch_callbacks_) cb();
    Job job;
    job.type = job_type;
    job.epoch = epoch;
    job.filename = job_type == Job::kValidation ? param_.val_data : param_.data_in;
    if (job.filename.empty()) return;

    job.num_parts = 1;
    std::vector<Job> jobs;
    for (int i = 0; i < job.num_parts; ++i) {
      job.part_idx = i;
      jobs.push_back(job);
    }
    job_tracker_->Add(jobs);

    while (job_tracker_->NumRemains() != 0) {
      Sleep();
      for (auto& cb : cont_callbacks_) cb();
    }
    for (auto& cb : epoch_callbacks_) cb();
  }



  struct BatchJob {
    int type;
    dmlc::data::RowBlockContainer<unsigned>* data;
    std::shared_ptr<std::vector<feaid_t>> feaids;
  };

  void ProcessFile(const Job& job) {
    Tracker<BatchJob> tracker;
    tracker.SetConsumer([this](const BatchJob& batch, const Callback& on_complete) {
        auto val = new std::vector<real_t>();
        auto val_siz = new std::vector<int>();

        auto pull_callback = [this, batch, val, val_siz, on_complete]() {
          // eval the objective,
          CHECK_NOTNULL(loss_)->InitData(batch.data->GetBlock(), *val, *val_siz);
          Progress prog; loss_->Evaluate(&prog);
          if (pmonitor_ == nullptr) {
            pmonitor_ = ProgressMonitor::Create();
          }
          pmonitor_->Add(prog);

          if (batch.type == Job::kTraining) {
            // calculate the gradients
            loss_->CalcGrad(val);

            // push the gradient, let the system delete val, val_siz. this task is
            // done only if the push is complete
            store_->Push(Store::kGradient,
                         batch.feaids,
                         std::shared_ptr<std::vector<real_t>>(val),
                         std::shared_ptr<std::vector<int>>(val_siz),
                         [on_complete]() { on_complete(); });
          } else {
            // save the prediction results
            if (batch.type == Job::kPrediction) {
              std::vector<real_t> pred;
              loss_->Predict(&pred);
              // for (real_t p : pred) CHECK_NOTNULL(pred_out) << p << "\n";
            }

            on_complete();
            delete val;
            delete val_siz;
          }
          delete batch.data;
        };
        store_->Pull(Store::kWeight, batch.feaids, val, val_siz, pull_callback);
      });

    int batch_size = 100;
    int shuffle = 0;
    float neg_sampling = 1;
    BatchIter reader(
        job.filename, param_.data_format, job.part_idx, job.num_parts,
        batch_size, shuffle, neg_sampling);
    while (reader.Next()) {
      // map feature id into continous index
      BatchJob batch;
      batch.type = job.type;
      batch.data = new dmlc::data::RowBlockContainer<unsigned>();
      batch.feaids = std::make_shared<std::vector<feaid_t>>();
      auto feacnt = std::make_shared<std::vector<real_t>>();

      bool push_cnt =
          job.type == Job::kTraining && job.epoch == 0;

      Localizer lc(-1, 2);
      lc.Compact(reader.Value(), batch.data, batch.feaids.get(),
                 push_cnt ? feacnt.get() : nullptr);

      if (push_cnt) {
        auto empty = std::make_shared<std::vector<int>>();
        store_->Wait(store_->Push(Store::kFeaCount, batch.feaids, feacnt, empty));
      }

      while (tracker.NumRemains() > 1) Sleep(10);

      tracker.Add({batch});
    }

    tracker.Wait();
    // while (tracker.NumRemains() > 0) Sleep(10);
  }

 private:
  /** \brief the model store*/
  Store* store_;
  /** \brief the loss*/
  Loss* loss_;
  /** \brief parameters */
  SGDLearnerParam param_;
  ProgressPrinter pprinter_;
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_SGD_LEARNER_H_
