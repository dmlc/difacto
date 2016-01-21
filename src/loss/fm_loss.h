/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_FM_LOSS_H_
#define DIFACTO_LOSS_FM_LOSS_H_
#include <vector>
#include <cmath>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "difacto/loss.h"
#include "common/spmv.h"
#include "common/spmm.h"
#include "./fm_loss_utils.h"
#include "./logit_loss.h"
namespace difacto {
/**
 * \brief parameters for FM loss
 */
struct FMLossParam : public dmlc::Parameter<FMLossParam> {
  /**
   * \brief the embedding dimension
   */
  int V_dim;
  // /**
  //  * \brief the probability to set gradient of :math:`V` to 0. In default is 0
  //  */
  // float V_dropout;
  // /**
  //  * \brief project the gradient of :math:`V` into :math:`[-c c]`. No projection
  //  * in default
  //  */
  // float V_grad_clipping;
  // /**
  //  * \brief normalize the gradient of :math:`V`. No normalizationin in default
  //  */
  // float V_grad_normalization;
  DMLC_DECLARE_PARAMETER(FMLossParam) {
    DMLC_DECLARE_FIELD(V_dim).set_range(0, 10000);
    // DMLC_DECLARE_FIELD(V_dropout).set_range(0, 1).set_default(0);
    // DMLC_DECLARE_FIELD(V_grad_clipping).set_range(0, 1000.0).set_default(0);
    // DMLC_DECLARE_FIELD(V_grad_normalization).set_range(0, 1000.0).set_default(0);
  }
};
/**
 * \brief the factonization machine loss
 * :math:`f(x) = \langle w, x \rangle + \frac{1}{2} \|V x\|_2^2 - \sum_{i=1}^d x_i^2 \|V_i\|^2_2`
 */
class FMLoss : public Loss {
 public:
  FMLoss() {}
  virtual ~FMLoss() {}

  KWArgs Init(const KWArgs& kwargs) override {
    return param_.InitAllowUnknown(kwargs);
  }
  /**
   * \brief perform prediction
   *
   *  pred = X * w + .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
   *
   * where
   * - sum(A, 2) : sum the rows of A
   * - .* : elemenetal-wise times
   *
   * @param data the data
   * @param param input parameters
   * - param[0], real_t vector, the weights
   * - param[1], int vector, the weight positions
   * @param pred predict output, should be pre-allocated
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    // pred = X * w
    SArray<real_t> w(param[0]);
    SArray<int> w_pos(param[1]);
    SpMV::Times(data, w, pred, nthreads_, w_pos, {});

    int V_dim = param_.V_dim;
    if (V_dim == 0) return;

    SArray<real_t> V(param[0]);
    SArray<int> V_pos = GetVPos(SArray<int>(param[1]), V.size());

    // XV_ = X*V
    XV_.resize(data.size * V_dim);
    SpMM::Times(data, V, &XV_, V_dim, nthreads_, V_pos);

    // XX = X.*X
    auto XX = data;
    if (XX.value) {
      XX_.CopyFrom(XX.value+XX.offset[0], XX.offset[XX.size] - XX.offset[0]);
      for (real_t& v : XX_) v *= v;
      XX.value = XX_.data();
    }

    // VV = V*V
    SArray<real_t> VV(V.size());
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < V_pos.size(); ++i) {
      int p = V_pos[i]; if (p < 0) continue;
      for (int j = 0; j < V_dim; ++j) VV[p+j] *= V[p+j];
    }

    // XXVV = XX*VV
    SArray<real_t> XXVV(XV_.size());
    SpMM::Times(XX, VV, &XXVV, V_dim, nthreads_, V_pos);

    // py += .5 * sum((V.XV).^2 - xxvv)
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < pred->size(); ++i) {
      real_t* t = XV_.data() + i * V_dim;
      real_t* tt = XXVV.data() + i * V_dim;
      real_t s = 0;
      for (int j = 0; j < V_dim; ++j) s += t[j] * t[j] - tt[j];
      (*pred)[i] += .5 * s;
    }
  }

  /*!
   * \brief compute the gradients
   *
   *   p = - y ./ (1 + exp (y .* pred));
   *   grad_w = X' * p;
   *   grad_u = X' * diag(p) * X * V  - diag((X.*X)'*p) * V
   *
   * @param data the data
   * @param param input parameters
   * - param[0], real_t vector, the predict output
   * - param[1], real_t vector, the weights
   * - param[2], int vector, the gradient positions
   * @param grad the results
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    // p = ...
    SArray<real_t> p; p.CopyFrom(SArray<real_t>(param[0]));
    CHECK_EQ(p.size(), data.size);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = data.label[i] > 0 ? 1 : -1;
      p[i] = - y / (1 + std::exp(y * p[i]));
    }

    // grad_w = ...
    SArray<int> w_pos(param[2]);
    SpMV::TransTimes(data, p, grad, nthreads_, {}, w_pos);

    // grad_u = ...
    int V_dim = param_.V_dim;
    if (V_dim == 0) return;
    SArray<real_t> V(param[0]);
    SArray<int> V_pos = GetVPos(SArray<int>(param[2]), V.size());

    // XXp = (X.*X)'*p
    auto XX = data;
    if (XX.value) {
      CHECK_EQ(XX_.size(), XX.offset[XX.size] - XX.offset[0]);
      XX.value = XX_.data();
    }
    SArray<real_t> XXp(V_pos.size());
    SpMM::TransTimes(XX, p, &XXp, V_dim, nthreads_);

    // grad_u = - diag(XXp) * V,
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < V_pos.size(); ++i) {
      int p = V_pos[i]; if (p < 0) continue;
      for (int j = 0; j < V_dim; ++j) {
        (*grad)[p+j] = - V[p+j] * XXp[i];
      }
    }

    // XV_ = diag(p) * X * V
    CHECK_EQ(XV_.size(), data.size * V_dim);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < p.size(); ++i) {
      for (int j = 0; j < V_dim; ++j) XV_[i*V_dim+j] *= p[i];
    }

    // grad_u += X' * diag(p) * X * V
    SpMM::TransTimes(data, XV_, grad, V_dim, nthreads_, {}, V_pos);
  }

 private:
  SArray<int> GetVPos(const SArray<int>& pos, int value_size) {
    SArray<int> V_pos;
    V_pos.CopyFrom(SArray<int>(pos));
    for (int& p : V_pos) p = p < 0 ? -1 : p + 1;
    int& back = V_pos[V_pos.size()-1];
    if (back + param_.V_dim > static_cast<int>(value_size)) {
      back = -1;
    }
    return V_pos;
  }
  SArray<real_t> XV_, XX_;
  FMLossParam param_;
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_FM_LOSS_H_
