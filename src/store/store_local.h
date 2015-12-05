#ifndef DIFACTO_STORE_STORE_LOCAL_H_
#define DIFACTO_STORE_STORE_LOCAL_H_
#include "difacto/model.h"
#include "difacto/model_sync.h"
namespace difacto {


struct StoreLocalParam : public dmlc::Parameter<StoreLocal> {
  /** \brief type of model, default is sgd */
  std::string model;
};
/**
 * \brief model sync within a machine
 */
class StoreLocal {
 public:
  StoreLocal() : model_(nullptr) { }
  virtual ~StoreLocal() { }

  KWArgs Init(const KWArgs& kwargs) {

  }

 private:
  Model* model_;
};
}  // namespace difacto
#endif  // DIFACTO_STORE_STORE_LOCAL_H_
