/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/store.h"
#include "./store_local.h"
namespace difacto {

Store* Store::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new StoreLocal();
  }
}

}  // namespace difacto
