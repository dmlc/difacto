/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/reporter.h"
#include "./local_reporter.h"
namespace difacto {

Reporter* Reporter::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new LocalReporter();
  }
}

}  // namespace difacto
