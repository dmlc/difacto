#include "difacto/store.h"
#include "./store_local.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(StoreLocalParam);

Store* Store::Create(const std::string& type) {
  if (type == "local") {
    return new StoreLocal();
  } else {
    LOG(FATAL) << "unknown store type " << type;
  }
  return nullptr;
}

}  // namespace difacto
