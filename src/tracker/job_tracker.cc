#include "difacto/job.h"
#include "./job_tracker_local.h"
namespace difacto {

DMLC_REGISTER_PARAMETER(Job);

JobTracker* JobTracker::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new JobTrackerLocal();
  }
}

}  // namespace difacto
