#ifndef DIFACTO_PROGRESS_H_
#define DIFACTO_PROGRESS_H_
#include <vector>
#include <string>
#include "./base.h"
namespace difacto {
/**
 * \brief the training progress
 */
struct Progress {
 public:
  Progress() : data(8) { }

  /**
   * \brief merge from another progress
   */
  void Merge(const Progress& other) {
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] += other.data[i];
    }
  }

  /**
   * \brief accessors & mutators
   */
  real_t& objv() { return data[0]; }
  real_t objv() const { return data[0]; }

  real_t& auc() { return data[1]; }
  real_t auc() const { return data[1]; }

  real_t& objv_w() { return data[2]; }
  real_t objv_w() const { return data[2]; }

  real_t& copc() { return data[3]; }
  real_t copc() const { return data[3]; }

  real_t& count() { return data[4]; }
  real_t count() const { return data[4]; }

  real_t& new_ex() { return data[5]; }
  real_t new_ex() const { return data[5]; }

  real_t& new_w() { return data[6]; }
  real_t new_w() const { return data[6]; }

  real_t& new_V() { return data[7]; }
  real_t new_V() const { return data[7]; }

  /**
   * \brief the actual data
   */
  std::vector<real_t> data;
};

class ProgressPrinter {
 public:
  /**
   * \brief return a readable header string
   * \parram show_V whether or not show V
   */
  std::string Head(bool show_V) {
    if (show_V) {
      return "  ttl #ex   inc #ex |  |w|_0  logloss_w |   |V|_0    logloss    AUC";
    } else {
      return "  ttl #ex   inc #ex |  |w|_0  logloss    logloss    AUC";
    }
  }

  /**
   * \brief return a readable string
   * \param show_V whether or not show V
   */
  std::string Body(const Progress& cur, bool show_V) {
    if (cur.new_ex() == prev_.new_ex()) return "";
    char buf[256];

    Progress diff;
    for (size_t i = 0; i < cur.data.size(); ++i) {
      diff.data[i] = cur.data[i] - prev_.data[i];
    }
    if (show_V) {
      snprintf(buf, 256, "%9.4g  %7.2g | %9.4g  %6.4f | %9.4g  %7.5f  %7.5f ",
               cur.new_ex(), diff.new_ex(),
               cur.new_w(), diff.objv_w() / diff.new_ex(),
               cur.new_V(), diff.objv() / diff.new_ex(),
               diff.auc() / diff.count());
    } else {
      snprintf(buf, 256, "%9.4g  %7.2g | %9.4g  %6.4f | %9.4g  %7.5f  %7.5f ",
               cur.new_ex(), diff.new_ex(),
               cur.new_w(), diff.objv() / diff.new_ex(),
               diff.auc() / diff.count());
    }
    prev_.data = cur.data;
    return std::string(buf);
  }
 private:
  Progress prev_;
};


}  // namespace difacto
#endif /* DIFACTO_PROGRESS_H_ */
