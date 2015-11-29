#pragma once
#include <stdlib.h>
#include <chrono>
// #include "model-inl.h"
// #include "loss.h"
namespace difacto {

/*!
 * \brief use float as the weight and gradient type
 */
typedef float real_t;

class DiFacto {
 public:
  DiFacto() { }
  ~DiFacto() { }

  /**
   * \brief Init difacto
   *
   * \param config the protobuf string of Config
   */
  void Init(const std::string& config) {
    // init config
    CHECK(conf_.ParseFromString(config));
    local_ = conf_.task() != "dist_train";

    // init job tracker
    tracker_ = JobTracker::Create(local_ ? "local" : "dist");
    using namespace std::placeholders;
    tracker_->SetConsumer(std::bind(&DiFacto::Process, this, _1));

    // init model and model_sync
    char* role_c = getenv("DMLC_ROLE");
    role_ = std::string(role_c, strlen(role_c));
    if (local_ || role_ == "server") {
      model_ = Model<real_t>::Create("sgd");
    }
    if (local_ || role_ == "worker") {
      model_sync_ = ModelSync<real_t>::Create(local_ ? "local" : "dist");
    }
  }

  /** \brief the callback function type */
  typedef std::function<void()> Callback;

  void AddEpochCallback(const Callback& callback);

  void AddContCallback(const Callback& callback);

  /**
   * \brief Run difacto
   */
  void Run() {
    if (local_ || role == "scheduler") {
      RunScheduler();
    } else {
      tracker_->Wait();
    }
  }

  void Stop() {

  }

  const std::vector<double>& GetProgress() const {}

 private:
  void RunScheduler() {
    if (!local_) {
      printf("Connected %d servers and %d workers\n",
             ps::NodeInfo::NumServers(), ps::NodeInfo::NumWorkers());
    }

    int cur_epoch = 0;

    if (predict) {
      RunEpoch(cur_epoch, Job::PREDICTION);
      return;
    }

    // train
    for (; cur_epoch < conf_.max_num_epochs(); ++ cur_epoch) {
      RunEpoch(cur_epoch, Job::TRAINING);
      RunEpoch(cur_epoch, Job::VALIDATION);
      for (auto& cb : epoch_callbacks_) cb();
    }

  }

  void RunEpoch(int epoch, Job::Type type) {
    Job job;
    job.type = job;
    job.epoch = epoch;
    job.filename = type == Job::VALIDATION ? conf_.val_data() : conf_.data_in();
    if (job.filename.empty()) return;
    job.num_parts = 100;

    std::vector<Job> jobs;
    for (int i = 0; i < job.num_parts; ++i) {
      job.part_idx = i;
      jobs.push_back(job);
    }
    CHECK_NOTNULL(tracker_)->Add(jobs);

    while (tracker_->NumRemains() != 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      for (auto& cb : cont_callbacks_) cb();
    }
  }

  /** \brief process a workload */
  void Process(const Job& job) {
    if (job.type == Job::SAVE_MODEL) {
      dmlc::Stream* fo;
      CHECK_NOTNULL(model_)->Save(fo);
    } else if (job.type == Job::LOAD_MODEL) {
      dmlc::Stream* fi;
      CHECK_NOTNULL(model_)->Load(fi);
    } else {
      ProcessFile(job);
    }
  }

  void ProcessFile(const Job& job) {
    int batch_size = 100;
    int shuffle = 0;
    float neg_sampling = 1;
    BatchIter<feaid_t> reader(
        job.filename, job.part_idx, job.num_parts, conf_.data_format(),
        batch_size, shuffle, neg_sampling);
    while (reader.Next()) {
      // map feature id into continous index
      auto batch = new dmlc::data::RowBlockContainer<unsigned>();
      auto feaids = std::make_shared<std::vector<feaid_t>>();
      auto feacnt = std::make_shared<std::vector<real_t>>();

      bool push_cnt =
          job.type == Job::TRAINING && job.epoch == 0 && conf_.V_dim() > 0;

      Localizer<feaid_t> lc(conf_.num_threads());
      lc.Localize(reader.Value(), batch, feaids.get(),
                  push_cnt ? feacnt.get() : nullptr);

      if (push_cnt) {
        auto empty = std::make_shared<std::vector<int>>();
        model_sync_->Wait(model_sync_->Push(feaids, feacnt, empty));
      }

      WaitBatch(10);

      ProcessBatch(job.type, batch->GetBlock(), feaids, nullptr, on_complete);
    }
    WaitBatch(1);
  }

  /**
   * \brief process a data batch
   *
   * @param type job type
   * @param batch data batch, whose feature has been mapped into continous index
   * @param feaids the according global feature ids
   * @param pred_out data stream for saving prediction results
   * @param on_complete call the callback when this workload is finished
   */
  void ProcessBatch(Job::Type type,
                    const dmlc::RowBlock<unsigned>& batch,
                    const std::shared_ptr<std::vector<feaid_t> >& feaids,
                    dmlc::Stream* pred_out,
                    Callback on_complete) {
    auto val = new std::vector<real_t>();
    auto val_siz = new std::vector<int>();

    auto pull_callback = [this, batch, feaids, val, val_siz]() {
      double start = GetTime();
      // eval the objective,
      Loss<real_t> loss = Loss<real_t>::Create("fm");
      loss->Init(batch, *val, *val_siz, conf_);
      Progress prog; loss->Evaluate(&prog);

      if (type == Job::TRAINING) {
        // calculate the gradients
        loss->CalcGrad(val);

        // push the gradient, let the system delete val, val_siz. this task is
        // done only if the push is complete
        model_sync_->Push(feaids,
                          std::shared_ptr<std::vector<real_t>>(val),
                          std::shared_ptr<std::vector<int>>(val_siz),
                          [on_complete]() { on_complete(); });
      } else {
        // save the prediction results
        if (type == Job::PREDICTION) {
          std::vector<real_t> pred;
          loss->Predict(&pred);
          CHECK_NOTNULL(pred_out);
          for (real_t p : pred) pred_out << p << "\n";
        }

        on_complete();
        delete val;
        delete val_siz;
      }
      workload_time_ += GetTime() - start;
    };
    model_sync_->Pull(feaids, val, val_siz, pull_callback);
  }

  Config conf_;
  JobTracker* tracker_;

  bool local_;
  std::string role_;


};

}  // namespace difacto
