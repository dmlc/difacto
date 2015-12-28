/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "./sgd_learner.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDLearnerParam);

Learner* Learner::Create(const std::string& type) {
  if (type == "sgd") {
    return new SGDLearner();
  } else {
    LOG(FATAL) << "unknown learner type";
  }
  return nullptr;
}

Learner::Learner() {
  pmonitor_ = nullptr;
}

Learner::~Learner() {
  delete pmonitor_;
}

KWArgs Learner::Init(const KWArgs& kwargs) {
  // init job tracker
  job_tracker_ = JobTracker::Create();
  auto remain = job_tracker_->Init(kwargs);
  using namespace std::placeholders;
  job_tracker_->SetConsumer(std::bind(&Learner::Process, this, _1));
  return remain;
}

}  // namespace difacto
