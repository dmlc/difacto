#include "difacto/model.h"
#include "./sgd.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDModelParam);
DMLC_REGISTER_PARAMETER(SGDUpdateParam);

Model* Model::Create(const std::string& type) {
  return nullptr;
}

}  // namespace difacto
