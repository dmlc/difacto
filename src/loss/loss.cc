#include "difacto/loss.h"
#include "./fm_loss.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(FMParam);

Loss* Loss::Create(const std::string& type) {
  if (type == "fm") {
    return new FMLoss();
  } else {
    LOG(FATAL) << "unknown loss type";
  }
  return nullptr;
}

}  // namespace difacto
