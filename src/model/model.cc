#include "difacto/model.h"
#include "./sgd.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDModelParam);
DMLC_REGISTER_PARAMETER(SGDOptimizerParam);

Model* Model::Create(const std::string& type) {
  if (type == "sgd") {
    return new SGDOptimizer();
  } else {
    LOG(FATAL) << "known model type " << type;
  }
  return nullptr;
}

}  // namespace difacto
