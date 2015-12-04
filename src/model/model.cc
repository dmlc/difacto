#include "difacto/model.h"
#include "./sgd.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(SGDModelParam);

Model* Model::Create(const std::string& type) {
  return nullptr;
}

}  // namespace difacto
