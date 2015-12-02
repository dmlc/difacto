#pragma once
#include "dmlc/data.h"
#include "difacto/base.h"
#include <string>
namespace difacto {

class BatchIter {
 public:
  BatchIter(const std::string& uri,
            unsigned part_index,
            unsigned num_parts,
            const std::string& format,
            unsigned batch_size,
            unsigned shuffle_size = 0,
            float neg_sampling = 1.0) {

  }


  bool Next(void) {
  }


  const dmlc::RowBlock<feaid_t>& Value(void) const {
  }
};

}  // namespace difacto
