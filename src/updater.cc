/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/updater.h"
// #include "./sgd/sgd_updater.h"
namespace difacto {

// DMLC_REGISTER_PARAMETER(SGDUpdaterParam);

Updater* Updater::Create(const std::string& type) {
  // if (type == "sgd") {
  //   return new SGDUpdater();
  // } else {
  //   LOG(FATAL) << "known updater type " << type;
  // }
  return nullptr;
}

}  // namespace difacto
