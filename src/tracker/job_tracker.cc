#include "difacto/job.h"
#include "./job_tracker_local.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(Job);

JobTracker* JobTracker::Create(const std::string& type) {
  if (type == "local") {
    return new JobTrackerLocal();
  } else {
    LOG(FATAL) << "unknown job tracker type " << type;
  }
  return NULL;
}

}  // namespace difacto
