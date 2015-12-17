/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_TRACKER_JOB_TRACKER_LOCAL_H_
#define DIFACTO_TRACKER_JOB_TRACKER_LOCAL_H_
#include <vector>
#include "difacto/job.h"
#include "./tracker.h"
namespace difacto {

class JobTrackerLocal : public JobTracker {
 public:
  JobTrackerLocal() : tracker_(nullptr) { }
  virtual ~JobTrackerLocal() { delete tracker_; }

  KWArgs Init(const KWArgs& kwargs) override {
    Init();
    return kwargs;
  }

  void Add(const std::vector<Job>& jobs) override {
    Init();
    tracker_->Add(jobs);
  }

  int NumRemains() override {
    return CHECK_NOTNULL(tracker_)->NumRemains();
  }

  void Clear() override {
    CHECK_NOTNULL(tracker_)->Clear();
  }

  void Stop() override {
    if (tracker_) {
      delete tracker_;
      tracker_ = nullptr;
    }
  }

  void Wait() override {
    Stop();
  }

  void SetConsumer(const Consumer& consumer) override {
    CHECK_NOTNULL(tracker_)->SetConsumer(
        [consumer](const Job& job, const Tracker<Job>::Callback& on_complete) {
          consumer(job);
          on_complete();
        });
  }

 private:
  inline void Init() {
    if (!tracker_) tracker_ = new Tracker<Job>();
  }

  Tracker<Job>* tracker_;
};

}  // namespace difacto
#endif  // DIFACTO_TRACKER_JOB_TRACKER_LOCAL_H_
