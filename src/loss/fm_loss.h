/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_FM_LOSS_H_
#define DIFACTO_LOSS_FM_LOSS_H_
#include <vector>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "difacto/loss.h"
#include "common/spmv.h"
#include "common/spmm.h"
#include "./fm_loss_utils.h"
namespace difacto {
/**
 * \brief parameters for FM loss
 */
struct FMLossParam : public dmlc::Parameter<FMLossParam> {
  /**
   * \brief the embedding dimension
   */
  int V_dim;
  /**
   * \brief the probability to set gradient of :math:`V` to 0. In default is 0
   */
  float V_dropout;
  /**
   * \brief project the gradient of :math:`V` into :math:`[-c c]`. No projection
   * in default
   */
  float V_grad_clipping;
  /**
   * \brief normalize the gradient of :math:`V`. No normalizationin default
   */
  float V_grad_normalization;
  /** \brief number of threads */
  int nthreads;
  DMLC_DECLARE_PARAMETER(FMLossParam) {
    DMLC_DECLARE_FIELD(V_dim).set_range(0, 10000);
    DMLC_DECLARE_FIELD(V_dropout).set_range(0, 1).set_default(0);
    DMLC_DECLARE_FIELD(V_grad_clipping).set_range(0, 1000.0).set_default(0);
    DMLC_DECLARE_FIELD(V_grad_normalization).set_range(0, 1000.0).set_default(0);
    DMLC_DECLARE_FIELD(nthreads).set_range(1, 20).set_default(2);
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
   * @data data the data
   * @param param the weights
   * - param[0], real_t, the weights
   * - param[1], int, the weight lenghs
   * @param pred the prediction
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    // init data
    SArray<real_t> weights(param[0]);
    SArray<int> weight_lens(param[1]);
    InitData(data, weights, weight_lens);
    int nt = param_.nthreads;
    int V_dim = param_.V_dim;
    auto X = data;
    pred->resize(X.size, 0.0);

    // pred = X * w
    SpMV::Times(X, w.value, pred, nt);

    // pred += .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
    if (V.value.size()) {
      // tmp = (X.*X)*(V.*V)
      SArray<real_t> vv; vv.CopyFrom(V.value);
      for (auto& v : vv) v *= v;
      CHECK_EQ(vv.size(), V.pos.size() * V_dim);
      SArray<real_t> xxvv(V.X.size * V_dim, 0.0);
      SpMM::Times(V.XX, vv, &xxvv, nt);

      // V.XV = X*V
      V.XV.resize(xxvv.size());
      SpMM::Times(V.X, V.value, &V.XV, nt);

      // py += .5 * sum((V.XV).^2 - xxvv)
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < pred->size(); ++i) {
        real_t* t = V.XV.data() + i * V_dim;
        real_t* tt = xxvv.data() + i * V_dim;
        real_t s = 0;
        for (int j = 0; j < V_dim; ++j) s += t[j] * t[j] - tt[j];
        (*pred)[i] += .5 * s;
      }
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
   * @param param the weights
   * - param[0], real_t, the weights
   * - param[1], int, the weight lenghs
   * - param[2], real_t, the prediction (results of \ref Predict)
   * @param grad the results
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    // init data
    SArray<real_t> weights(param[0]);
    SArray<int> weight_lens(param[1]);
    InitData(data, weights, weight_lens);
    int nt = param_.nthreads;
    SArray<real_t> p; p.CopyFrom(SArray<real_t>(param[2]));
    auto X = data;

    // p = ...
    CHECK_EQ(p.size(), X.size);
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < p.size(); ++i) {
      real_t y = X.label[i] > 0 ? 1 : -1;
      p[i] = - y/(1 + exp(y * p[i]));
    }

    // grad_w = ...
    SpMV::TransTimes(X, p, &w.value, nt);
    w.Save(grad);

    // grad_u = ...
    if (!V.value.empty()) {
      int dim = param_.V_dim;

      // xxp = (X.*X)'*p
      size_t m = V.pos.size();

      SArray<real_t> xxp(m, 0.0);
      SpMM::TransTimes(V.XX, p, &xxp, nt);

      // V = - diag(xxp) * V
      CHECK_EQ(V.value.size(), dim * m);
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < m; ++i) {
        real_t* v = V.value.data() + i * dim;
        for (int j = 0; j < dim; ++j) v[j] *= - xxp[i];
      }

      // V.XV = diag(p) * X * V
      size_t n = p.size();
      CHECK_EQ(V.XV.size(), n * dim);
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < n; ++i) {
        real_t* y = V.XV.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[j] *= p[i];
      }

      // V += X' * V.XV
      SpMM::TransTimes(V.X, V.XV, (real_t)1.0, V.value, &V.value, nt);

      // some preprocessing
      real_t gc = param_.V_grad_clipping;
      if (gc > 0) {
        for (real_t& g : V.value) g = g > gc ? gc : ( g < -gc ? -gc : g);
      }

      real_t dp = param_.V_dropout;
      if (dp > 0) {
        for (real_t& g : V.value) {
          if ((real_t)rand() / RAND_MAX > 1 - dp) g = 0;
        }
      }

      if (param_.V_grad_normalization) {
        real_t norm = 0;
        for (real_t g : V.value) norm += g * g;
        if (norm < 1e-10) return;
        norm = sqrt(norm);
        for (real_t& g : V.value) g = g / norm;
      }
      V.Save(grad);
    }
  }

 private:
  void InitData(const dmlc::RowBlock<unsigned>& data,
                const SArray<real_t>& weights,
                const SArray<int>& weight_lens) {
    if (data_inited_) return;
    w.Init(weights, weight_lens);
    V.Init(param_.V_dim, data, weights, weight_lens);
    data_inited_ = true;
  }

  fmloss::Linear w;
  fmloss::Embedding V;
  FMLossParam param_;
  bool data_inited_ = false;
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_FM_LOSS_H_
