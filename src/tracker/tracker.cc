/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/tracker.h"
#include "./local_tracker.h"
namespace difacto {

Tracker* Tracker::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new LocalTracker();
  }
}

}  // namespace difacto
