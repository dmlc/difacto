#ifndef DIFACTO_LOSS_BIN_CLASS_EVAL_H_
#define DIFACTO_LOSS_BIN_CLASS_EVAL_H_
#include "difacto/base.h"
#include <algorithm>
#include <dmlc/logging.h>
#include <dmlc/omp.h>
namespace difacto {

/**
 * \brief binary class evaluation
 */
class BinClassEval {
 public:
  /**
   * \brief create
   *
   * @param label label vector
   * @param predict predict vector
   * @param n length
   * @param nthreads num threads
   */
  BinClassEval(const real_t* const label,
               const real_t* const predict,
               size_t n, int nthreads)
      : label_(label), predict_(predict), size_(n), nt_(nthreads) { }
  ~BinClassEval() { }

  real_t AUC() {
    size_t n = size_;
    struct Entry { real_t label; real_t predict; };
    std::vector<Entry> buff(n);
    for (size_t i = 0; i < n; ++i) {
      buff[i].label = label_[i];
      buff[i].predict = predict_[i];
    }
    std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
        return a.predict < b.predict; });
    real_t area = 0, cum_tp = 0;
    for (size_t i = 0; i < n; ++i) {
      if (buff[i].label > 0) {
        cum_tp += 1;
      } else {
        area += cum_tp;
      }
    }
    if (cum_tp == 0 || cum_tp == n) return 1;
    area /= cum_tp * (n - cum_tp);
    return area < 0.5 ? 1 - area : area;
  }

  real_t Accuracy(real_t threshold) {
    real_t correct = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:correct) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      if ((label_[i] > 0 && predict_[i] > threshold) ||
          (label_[i] <= 0 && predict_[i] <= threshold))
        correct += 1;
    }
    real_t acc = correct / (real_t) n;
    return acc > 0.5 ? acc : 1 - acc;
  }

  real_t LogLoss() {
    real_t loss = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:loss) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      real_t y = label_[i] > 0;
      real_t p = 1 / (1 + exp(- predict_[i]));
      p = p < 1e-10 ? 1e-10 : p;
      loss += y * log(p) + (1 - y) * log(1 - p);
    }
    return - loss;
  }

  real_t LogitObjv() {
    real_t objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      real_t y = label_[i] > 0 ? 1 : -1;
      objv += log( 1 + exp( - y * predict_[i] ));
    }
    return objv;
  }

  real_t Copc(){
    real_t clk = 0;
    real_t clk_exp = 0.0;
#pragma omp parallel for reduction(+:clk,clk_exp) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      if (label_[i] > 0) clk += 1;
      clk_exp += 1.0 / ( 1.0 + exp( - predict_[i] ));
    }
    return clk / clk_exp;
  }

 private:
  real_t const* label_;
  real_t const* predict_;
  size_t size_;
  int nt_;
};

}  // namespace difacto
#endif /* DIFACTO_LOSS_BIN_CLASS_EVAL_H_ */
