/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_LEARNER_UTILS_H_
#define DIFACTO_COMMON_LEARNER_UTILS_H_
#include <utility>
#include <string>
#include <vector>
#include "difacto/tracker.h"
#include "dmlc/io.h"
#include "dmlc/memory_io.h"
namespace difacto {
/**
 * \brief send jobs to a node group and wait them finished.
 *
 * @param node_group
 * @param job_args
 * @param tracker
 * @param job_rets
 */
inline void SendJobAndWait(int node_group,
                           const std::string& job_args,
                           Tracker* tracker,
                           std::vector<real_t>* job_rets) {
  // set monitor
  Tracker::Monitor monitor = nullptr;
  if (job_rets != nullptr) {
    monitor = [job_rets](int node_id, const std::string& rets) {
      auto copy = rets; dmlc::Stream* ss = new dmlc::MemoryStringStream(&copy);
      std::vector<real_t> vec; ss->Read(&vec); delete ss;
      if (job_rets->empty()) {
        *job_rets = vec;
      } else {
        CHECK_EQ(job_rets->size(), vec.size());
        for (size_t i = 0; i < vec.size(); ++i) (*job_rets)[i] += vec[i];
      }
    };
  }
  tracker->SetMonitor(monitor);

  // sent job
  std::pair<int, std::string> job;
  job.first = node_group;
  job.second = job_args;
  tracker->Issue({job});

  // wait until finished
  while (tracker->NumRemains() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}  // namespace difacto
#endif  // DIFACTO_COMMON_LEARNER_UTILS_H_
