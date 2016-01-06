#ifndef _LOGIT_LOSS_DELTA_H_
#define _LOGIT_LOSS_DELTA_H_
#include <cmath>
#include "difacto/loss.h"
#include "difacto/sarray.h"
#include "common/range.h"
#include "dmlc/omp.h"
#include "dmlc/logging.h"
namespace difacto {

/**
 * \brief parameters for FM loss
 */
struct LogitLossDeltaParam : public dmlc::Parameter<LogitLossDeltaParam> {
  /** \brief if or not compute the diagnal hession */
  int compute_diag_hession;
  /** \brief whether or not compute the upper bound of the diagnal hession */
  int compute_upper_diag_hession;
  /** \brief number of threads */
  int num_threads;
  DMLC_DECLARE_PARAMETER(LogitLossDeltaParam) {
    DMLC_DECLARE_FIELD(compute_upper_diag_hession).set_range(0, 1).set_default(1);
    DMLC_DECLARE_FIELD(compute_diag_hession).set_range(0, 1).set_default(0);
    DMLC_DECLARE_FIELD(num_threads).set_range(1, 20).set_default(DEFAULT_NTHREADS);
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
 * Note: Once can use \ref FMLoss with V_dim = 0 for ordinary logitis loss, namely given
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
   * @param param parameters
   * - param[0], real_t, previous predict
   * - param[1], real_t, delta weight, namely new_w - old_w
   * - param[2], len_t, optional, weight length. param[2][i] is the length of
   *   delta_w[i], 0 means skip, if > 1, then only the first one will pick. and  sum(param[2]) = length(delta_w)
   * @param pred output prediction, should be pre-allocated, it is safe to set
   * it be param[0]
   *  pred = X * w + prev_pred
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    // prepare
    int psize = param.size();
    CHECK_GE(psize, 2);
    CHECK_LE(psize, 3);
    SArray<real_t> Xw(param[0]);
    SArray<real_t> w(param[1]);

    len_t* w_len = nullptr;
    if (psize == 3 && param[2].size()) {
      SArray<len_t> len(param[2]);
      CHECK_EQ(len.size(), data.size);
      size_t n = 0; for (auto i : len) n += i;
      CHECK_EQ(n, w.size());
      w_len = reinterpret_cast<len_t*>(param[2].data());
    }

    if (pred->empty()) {
      pred->resize(Xw.size());
    } else {
      CHECK_EQ(pred->size(), Xw.size());
    }

    // compute
    if (pred->data() != Xw.data()) {
      memcpy(pred->data(), Xw.data(), Xw.size() * sizeof(real_t));
    }
    TransTimes(data, w.data(), w_len, pred->size(), pred->data());

    // project to [-15, 15]
    for (size_t i = 0; i < pred->size(); ++i) {
      real_t& p = (*pred)[i];
      p = p > 15 ? 15 : (p < -15 ? -15 : p);
    }
  }

  /**
   * @param data X', the transpose of X
   * @param param parameters
   * - param[0], real_t, previous predict
   * - param[1], lent_t, optional. weight length.
   * - param[2], real_t, delta, optional. should be given if
   *   compute_upper_diag_hession is true
   * @param grad output gradient
   *
   * tau = 1 / (1 + exp(y .* (X * w)))
   * first order grad
   *    f'(w) =  - X' * (tau .* y)
   * diagnal second order grad :
   *    f''(w) = (X.*X)' * (tau .* (1-tau))
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    // prepare
    int psize = param.size();
    CHECK_GE(psize, 1);
    CHECK_LE(psize, 2);
    SArray<real_t> Xw(param[0]);

    len_t* len = nullptr;
    size_t grad_size = 0;
    if (psize > 1 && param[1].size()) {
      SArray<len_t> l(param[1]);
      CHECK_EQ(l.size(), data.size);
      for (auto i : l) grad_size += i;
      len = reinterpret_cast<len_t*>(param[1].data());
    } else {
      grad_size = data.size;
    }

    real_t* delta = nullptr;
    if (psize > 2 && param[2].size()) {
      CHECK_EQ(param[2].size(), grad_size * sizeof(real_t));
      delta = reinterpret_cast<real_t*>(param[2].data());
      CHECK_EQ(param_.compute_upper_diag_hession, 1);
    } else {
      CHECK_EQ(param_.compute_upper_diag_hession, 0);
    }

    if (param_.compute_diag_hession || param_.compute_upper_diag_hession) {
      grad_size *= 2;
    }
    if (grad->empty()) {
      grad->resize(grad_size);
    } else {
      CHECK_EQ(grad->size(), grad_size);
    }

    // compute
    CalcGrad(data, Xw.data(), Xw.size(), len, delta, grad->data());
  }

 private:
  void CalcGrad(const dmlc::RowBlock<unsigned>& X,
                real_t const* Xw,
                size_t Xw_size,
                len_t const* len,
                real_t const* delta,
                real_t* grad) {
    SArray<real_t> tau(Xw_size);
#pragma omp parallel for num_threads(param_.num_threads)
    for (size_t i = 0; i < Xw_size; ++i) {
      real_t y = X.label[i] > 0 ? 1 : -1;
      tau[i] = 1.0 / (1 + std::exp(y * Xw[i]));
    }
    bool comp_h = param_.compute_diag_hession != 0;
    bool comp_u = param_.compute_upper_diag_hession != 0;
    CHECK(!(comp_h && comp_u));
    size_t g_skip = (comp_h || comp_u) ? 2 : 1;

#pragma omp parallel num_threads(param_.num_threads)
    {
      Range rg = Range(0, X.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());
      size_t cur_pos = 0;
      if (len) {
        for (size_t i = 0; i < rg.begin; ++i) cur_pos += len[i];
      } else {
        cur_pos = rg.begin;
      }
      size_t next_pos = cur_pos;
      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (X.offset[i] == X.offset[i+1]) continue;
        if (len) {
          if (len[i] == 0) continue;
          next_pos += len[i];
        } else {
          ++next_pos;
        }
        real_t d = comp_u ? (X.value ? delta[cur_pos] : std::exp(delta[cur_pos])) : 0;
        real_t g = 0, gg = 0;
        for (size_t j = X.offset[i]; j < X.offset[i+1]; ++j) {
          auto k = X.index[j];
          real_t y = X.label[k] > 0 ? 1 : -1;
          real_t t = tau[k];
          if (X.value) {
            real_t v = X.value[j];
            g -= y * t * v;
            if (comp_h) gg += t * (1 - t) * v * v;
            if (comp_u) gg += std::min(t * (1 - t) * std::exp(abs(v)*d),
                                       static_cast<real_t>(.25)) * v * v;
          } else {
            g -= y * t;
            if (comp_h) gg += t * (1 - t);
            if (comp_u) gg += std::min(t * (1 - t) * d, static_cast<real_t>(.25));
          }
        }
        *(grad+cur_pos*g_skip) = g;
        if (comp_h || comp_u) *(grad+cur_pos*g_skip+1) = gg;
        cur_pos = next_pos;
      }
    }
  }

  /**
   * y += D^T * x, with optional length for x
   */
  void TransTimes(const dmlc::RowBlock<unsigned>& D,
                  real_t const* x, len_t const* x_len, size_t y_size, real_t* y) {
#pragma omp parallel num_threads(param_.num_threads)
    {
      Range rg = Range(0, y_size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      size_t x_pos = 0;
      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        real_t x_i;
        if (x_len) {
          if (x_len[i] == 0) continue;
          x_i = x[x_pos]; x_pos += x_len[i];
        } else {
          x_i = x[i];
        }
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i * D.value[j];
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i;
          }
        }
      }
    }
  }
  LogitLossDeltaParam param_;
};

} // namespace difacto
#endif  // _LOGIT_LOSS_DELTA_H_
