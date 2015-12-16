#include "difacto/store.h"
#include "./store_local.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(StoreLocalParam);

Store* Store::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new StoreLocal();
  }
}

}  // namespace difacto
