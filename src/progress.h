#ifndef DIFACTO_PROGRESS_H_
#define DIFACTO_PROGRESS_H_
#include <vector>
#include <string>
namespace difacto {

/**
 * \brief the learning progress
 */
struct Progress {
  Progress() : data(8) { }
  /**
   * \brief return the head messsage
   */
  static std::string HeadStr(bool has_V) {
    if (has_V) {
      return "  ttl #ex   inc #ex |  |w|_0  logloss_w |   |V|_0    logloss    AUC";
    } else {
      return "  ttl #ex   inc #ex |  |w|_0  logloss    logloss    AUC";
    }
  }
  /**
   * \brief return a readable message
   */
  std::string PrintStr(bool has_V) {
    ttl_ex += new_ex();
    nnz_w += new_w();
    nnz_V += new_V();
    if (new_ex() == 0) return "";
    char buf[256];
    if (has_V) {
      snprintf(buf, 256, "%9.4g  %7.2g | %9.4g  %6.4lf | %9.4g  %7.5lf  %7.5lf ",
               ttl_ex, new_ex(), nnz_w, objv_w() / new_ex(), nnz_V,
               objv() / new_ex(),  auc() / count());
    } else {
      snprintf(buf, 256, "%9.4g  %7.2g | %9.4g  %6.4lf | %9.4g  %7.5lf  %7.5lf ",
               ttl_ex, new_ex(), nnz_w, objv_w() / new_ex(), nnz_V,
               objv() / new_ex(),  auc() / count());
    }
    return std::string(buf);
  }

  /// accessors & mutators
  real_t& objv() { return data[0]; }
  real_t& auc() { return data[1]; }
  real_t& objv_w() { return data[2]; }
  real_t& copc() { return data[3]; }

  real_t& count() { return data[4]; }
  real_t& new_ex() { return data[5]; }
  real_t& new_w() { return data[6]; }
  real_t& new_V() { return data[7]; }

  real_t objv() const { return data[0]; }
  real_t new_ex() const { return data[5]; }

  std::vector<real_t> data;
  real_t ttl_ex = 0, nnz_w = 0, nnz_V;
};

}  // namespace difacto

#endif /* DIFACTO_PROGRESS_H_ */
