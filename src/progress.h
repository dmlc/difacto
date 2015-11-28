#pragma once
#include <vector>
#include <string>
namespace difacto {

/**
 * \brief the progress of difactor
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
  double& objv() { return data[0]; }
  double& auc() { return data[1]; }
  double& objv_w() { return data[2]; }
  double& copc() { return data[3]; }

  double& count() { return data[4]; }
  double& new_ex() { return data[5]; }
  double& new_w() { return data[6]; }
  double& new_V() { return data[7]; }

  double objv() const { return data[0]; }
  double new_ex() const { return data[5]; }

  std::vector<double> data;
  double ttl_ex = 0, nnz_w = 0, nnz_V;
};

}  // namespace difacto
