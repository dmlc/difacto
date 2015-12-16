#include "difacto/progress.h"
#include "progress/progress_monitor_local.h"
namespace difacto {

std::mutex ProgressMonitorLocal::mu_;
Progress ProgressMonitorLocal::prog_;

std::string ProgressPrinter::Body(const Progress& cur) {
  if (cur.new_ex() == prev_.new_ex()) return "";
  char buf[256];

  Progress diff;
  for (size_t i = 0; i < cur.data.size(); ++i) {
    diff.data[i] = cur.data[i] - prev_.data[i];
  }
  snprintf(buf, 256, "%8.3g  %9.4g %9.4g  %6.4f %7.5f %6.4f %6.4f",
           diff.new_ex(),
           cur.new_w(),
           cur.new_V(),
           diff.objv_w() / diff.new_ex(),
           diff.objv() / diff.new_ex(),
           diff.acc() / diff.count(),
           diff.auc() / diff.count());
  prev_.data = cur.data;
  return std::string(buf);
}

ProgressMonitor* ProgressMonitor::Create() {
  if (IsDistributed()) {
    LOG(FATAL) << "not implemented";
    return nullptr;
  } else {
    return new ProgressMonitorLocal();
  }
}

}  // namespace difacto
