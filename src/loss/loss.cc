/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/loss.h"
#include "./fm_loss.h"
#include "./logit_loss_delta.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(FMLossParam);
DMLC_REGISTER_PARAMETER(LogitLossDeltaParam);

Loss* Loss::Create(const std::string& type) {
  if (type == "fm") {
    return new FMLoss();
  } else {
    LOG(FATAL) << "unknown loss type";
  }
  return nullptr;
}

}  // namespace difacto
