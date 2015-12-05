/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "./sgd.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDLearnerParam);

Learner* Learner::Create(const std::string& type) {
  if (type == "sgd") {
    return new SGDLearner();
  } else {
    LOG(FATAL) << "known learner type " << type;
  }
  return nullptr;
}

}  // namespace difacto
