/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_REPORTER_LOCAL_REPORTER_H_
#define DIFACTO_REPORTER_LOCAL_REPORTER_H_
namespace difacto {

class LocalReporter : public Reporter {
 public:
  LocalReporter() { }
  virtual ~LocalReporter() { }

  KWArgs Init(const KWArgs& kwargs) override { return kwargs; }

  void SetMonitor(const Monitor& monitor) override {
    monitor_ = monitor;
  }

  int Report(const std::string& report) {
    monitor_(-1, report); return 0;
  }

  void Wait(int timestamp) { }

 private:
  Monitor monitor_;
};
}  // namespace difacto
#endif  // DIFACTO_REPORTER_LOCAL_REPORTER_H_
