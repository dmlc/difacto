/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/loss.h"
#include "./fm_loss.h"
#include "./logit_loss_delta.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(FMLossParam);
DMLC_REGISTER_PARAMETER(LogitLossDeltaParam);

Loss* Loss::Create(const std::string& type, int nthreads) {
  Loss* loss = nullptr;
  if (type == "fm") {
    loss = new FMLoss();
  } else {
    LOG(FATAL) << "unknown loss type";
  }
  loss->set_nthreads(nthreads);
  return loss;
}

}  // namespace difacto
