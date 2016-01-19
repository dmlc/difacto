/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_LOGIT_LOSS_H_
#define DIFACTO_LOSS_LOGIT_LOSS_H_
#include <vector>
#include <cmath>
#include "difacto/base.h"
#include "difacto/loss.h"
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "common/spmv.h"
namespace difacto {

/**
 * \brief the logistic loss
 *
 * :math:`\ell(x,y,w) =  log(1 + exp(- y <w, x>))`
 *
 */
class LogitLoss : public Loss {
 public:
  LogitLoss() {}
  virtual ~LogitLoss() {}

  KWArgs Init(const KWArgs& kwargs) override {
    return kwargs;
  }

  /**
   * \brief perform prediction
   *
   *  pred += X * w
   *
   * @param data the data X
   * @param param input parameters
   * - param[0], real_t vector, the weights
   * - param[1], optional int vector, the weight positions
   * @param pred predict output, should be pre-allocated
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    int psize = param.size();
    CHECK_GE(psize, 1); CHECK_LE(psize, 2);
    SArray<real_t> w(param[0]);
    SArray<int> w_pos = psize == 2 ? param[1] : {};
    SpMV::Times(data, w, pred, w_pos, {}, nthreads_);
  }

  /*!
   * \brief compute the gradients
   *
   *   p = - y ./ (1 + exp (y .* pred));
   *   grad += X' * p;
   *
   * @param data the data X
   * @param param input parameters
   * - param[0], real_t vector, the predict output
   * - param[1], optional int vector, the gradient positions
   * @param grad the results, should be pre-allocated
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    int psize = param.size();
    CHECK_GE(psize, 1);
    CHECK_LE(psize, 2);
    SArray<real_t> p; p.CopyFrom(param[0]);
    SArray<int> grad_pos = psize == 2 ? param[1] : {};

    // p = ...
    CHECK_NOTNULL(X.label);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = X.label[i] > 0 ? 1 : -1;
      p[i] = - y / (1 + std::exp(y * p[i]));
    }

    // grad += ...
    SpMV::TransTimes(data, p, grad, {}, grad_pos, nthreads_);
  }
};
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_LOGIT_LOSS_H_
