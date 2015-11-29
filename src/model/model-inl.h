#pragma once
#include "model.h"
#include "sgd.h"
#include <limits>

namespace difacto {

template <typename T>
static Model<T>* Model<T>::Create(const Config& conf) {
  if (conf.algo() == "sgd") {

    // FeaID end_id = std::numeric_limits<FeaID>::max();
  } else {
    LOG(FATAL) << "unknow algo: " << conf.algo();
  }

  return nullptr;
}

}  // namespace difacto
