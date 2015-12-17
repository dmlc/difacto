/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_PROGRESS_PROGRESS_MONITOR_LOCAL_H_
#define DIFACTO_PROGRESS_PROGRESS_MONITOR_LOCAL_H_
#include <mutex>
#include "difacto/progress.h"
namespace difacto {

class ProgressMonitorLocal : public ProgressMonitor {
 public:
  virtual ~ProgressMonitorLocal() { }
  void Add(const Progress& prog) override {
    std::lock_guard<std::mutex> lk(mu_);
    prog_.Merge(prog);
  }

  void Get(Progress* prog) override {
    std::lock_guard<std::mutex> lk(mu_);
    *CHECK_NOTNULL(prog) = prog_;
  }
 private:
  static std::mutex mu_;
  static Progress prog_;
};
}  // namespace difacto
#endif  // DIFACTO_PROGRESS_PROGRESS_MONITOR_LOCAL_H_
