/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_LOGIT_LOSS_DELTA_H_
#define DIFACTO_LOSS_LOGIT_LOSS_DELTA_H_
#include <cmath>
#include <vector>
#include "difacto/loss.h"
#include "difacto/sarray.h"
#include "common/range.h"
#include "common/spmv.h"
#include "dmlc/omp.h"
#include "dmlc/logging.h"
namespace difacto {

/**
 * \brief parameters for \ref LogitLossDelta
 */
struct LogitLossDeltaParam : public dmlc::Parameter<LogitLossDeltaParam> {
  /**
   * \brief if or not compute the hession matrix
   * 0 : no
   * 1 : diagnal hession matrix
   * 2 : the upper bound of the diagnal hession
   */
  int compute_hession;
  DMLC_DECLARE_PARAMETER(LogitLossDeltaParam) {
    DMLC_DECLARE_FIELD(compute_hession).set_range(0, 2).set_default(1);
  }
};

/**
 * \brief the logistic loss, specialized for block coordinate descent
 *
 * :math:`\ell(x,y,w) =  log(1 + exp(- y <w, x>))`
 *
 * \ref LogitLossDelta is feeded with X' (the tranpose of X, in row-major
 * format) and delta w each time, and is able to compute the second order
 * gradients.
 *
 * Note: One can use \ref LogitLoss for ordinary logitis loss, namely given
 * X and w each time.
 */
class LogitLossDelta : public Loss {
 public:
  /** \brief constructor */
  LogitLossDelta() { }
  /** \brief deconstructor */
  virtual ~LogitLossDelta() { }
  KWArgs Init(const KWArgs& kwargs) override {
    return param_.InitAllowUnknown(kwargs);
  }
  /**
   * @param data X', the transpose of X
   *
   *  pred += X * delta_w
   *
   * @param param input parameters
   * - param[1], real_t vector, the delta weight, namely new_w - old_w
   * - param[2], optional int vector, the weight positions
   * @param pred predict output, should be pre-allocated
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    int psize = param.size();
    CHECK_GE(psize, 1); CHECK_LE(psize, 2);
    SArray<real_t> delta_w(param[0]);
    SArray<int> w_pos = psize == 2 ? SArray<int>(param[1]) : SArray<int>();
    SpMV::TransTimes(data, delta_w, pred, nthreads_, w_pos, {});
  }

  /**
   * \brief compute the gradients
   *
   * tau = 1 / (1 + exp(y .* pred))
   * first order grad
   *    f'(w) =  - X' * (tau .* y)
   * diagnal second order grad :
   *    f''(w) = (X.*X)' * (tau .* (1-tau))
   *
   * @param data X', the transpose of X
   * @param param input parameters
   * - param[0], real_t vector, the predict output
   * - param[1], optional int vector, the gradient positions
   * - param[2], optional real_t vectorreal_t, the delta needed if
   *   compute_diag_hession == 2
   * @param grad gradient output, should be preallocated
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    int psize = param.size();
    CHECK_GE(psize, 1);
    CHECK_LE(psize, 3);
    if (grad->empty()) return;

    // p = ...
    SArray<real_t> p; p.CopyFrom(SArray<real_t>(param[0]));
    CHECK_NOTNULL(data.label);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = data.label[i] > 0 ? 1 : -1;
      p[i] = - y / (1 + std::exp(y * p[i]));
    }

    // grad = ...
    SArray<int> grad_pos = psize > 1 ? SArray<int>(param[1]) : SArray<int>();
    if (param_.compute_hession != 0) CHECK(!grad_pos.empty());
    SpMV::Times(data, p, grad, nthreads_, {}, grad_pos);
    if (param_.compute_hession == 0) return;

    // h = ...
    SArray<int> h_pos; h_pos.CopyFrom(grad_pos);
    for (size_t i = 0; i < h_pos.size(); ++i) {
      if (h_pos[i] >= 0) ++h_pos[i];
    }

    // compute X .* X
    dmlc::RowBlock<unsigned> XX = data;
    SArray<real_t> xx_value;
    if (data.value) {
      xx_value.resize(data.offset[data.size]);
      for (size_t i = data.offset[0]; i < data.offset[data.size]; ++i) {
        xx_value[i] = data.value[i] * data.value[i];
      }
      XX.value = xx_value.data();
    }

    // p = tau * (1 - tau)
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = data.label[i] > 0 ? 1 : -1;
      p[i] = - p[i] * (y + p[i]);
    }

    if (param_.compute_hession == 1) {
      SpMV::Times(XX, p, grad, nthreads_, {}, h_pos);
    } else if (param_.compute_hession == 2) {
      LOG(FATAL) << "...";
      CHECK_EQ(psize, 3);
      SArray<real_t> delta(param[2]);
      // TODO(mli)
    } else {
      LOG(FATAL) << "...";
    }
  }

 private:
  LogitLossDeltaParam param_;
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_LOGIT_LOSS_DELTA_H_
