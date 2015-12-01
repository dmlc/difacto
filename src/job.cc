#include "difacto/job.h"

namespace difacto {

DMLC_REGISTER_PARAMETER(Job);

JobTracker* JobTracker::Create(const std::string& type) {
  return NULL;
}

}  // namespace difacto
