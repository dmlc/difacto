/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/updater.h"
#include "./sgd/sgd_updater.h"
#include "./bcd/bcd_updater.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDUpdaterParam);
// DMLC_REGISTER_PARAMETER(BCDUpdaterParam);

Updater* Updater::Create(const std::string& type) {
  if (type == "sgd") {
    return new SGDUpdater();
  } else if (type == "bcd") {
    return new BCDUpdater();
  } else {
    LOG(FATAL) << "known updater type " << type;
  }
  return nullptr;
}

}  // namespace difacto
