/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_FM_H_
#define DIFACTO_LOSS_FM_H_
#include <vector>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "dmlc/io.h"
#include "difacto/loss.h"
#include "common/spmv.h"
#include "common/spmm.h"
#include "difacto/progress.h"
#include "./bin_class_eval.h"
namespace difacto {

/**
 * \brief parameters for FM
 */
struct FMParam : public dmlc::Parameter<FMParam> {
  /** \brief the embedding dimension */
  int V_dim;
  /**
   * \brief the probability to set gradient of :math:`V` to 0
   * 0 in default
   */
  float V_dropout;
  /**
   * \brief project the gradient of :math:`V` into :math:`[-c c]`.
   * No in default
   */
  float V_grad_clipping;
  /**
   * \brief normalize the gradient of :math:`V`
   * No in default
   */
  float V_grad_normalization;
  /** \brief number of threads */
  int nthreads;
  DMLC_DECLARE_PARAMETER(FMParam) {
    DMLC_DECLARE_FIELD(V_dim).set_range(0, 10000);
    DMLC_DECLARE_FIELD(V_dropout).set_range(0, 1).set_default(0);
    DMLC_DECLARE_FIELD(V_grad_clipping).set_range(0, 1000.0).set_default(0);
    DMLC_DECLARE_FIELD(V_grad_normalization).set_range(0, 1000.0).set_default(0);
    DMLC_DECLARE_FIELD(nthreads).set_range(1, 20).set_default(2);
  }
};

/**
 * \brief the FM loss
 */
class FMLoss : public Loss {
 public:
  FMLoss() {}
  virtual ~FMLoss() {}

  KWArgs Init(const KWArgs& kwargs) override {
    return param_.InitAllowUnknown(kwargs);
  }

  /**
   * note: do not clear \ref data before call \ref Evaluate or \ref CalcGrad
   */
  void InitData(const dmlc::RowBlock<unsigned>& data,
                const std::vector<real_t>& weights,
                const std::vector<int>& weight_lens) override {
    w.Load(data, weights, weight_lens);
    V.Load(param_.V_dim, data, weights, weight_lens);
  }

  void Clear() override {
    w.Clear(); V.Clear();
  }

  /**
   * \brief evaluate the progress
   *
   *  py = X * w + .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
   *
   * where
   * - sum(A, 2) : sum the rows of A
   * - .* : elemenetal-wise times
   */
  void Evaluate(Progress* prog) {
    int nt = param_.nthreads;
    int V_dim = param_.V_dim;
    py_.resize(w.X.size);
    BinClassEval eval(w.X.label, py_.data(), py_.size(), nt);

    // py = X * w
    SpMV::Times(w.X, w.weight, &py_, nt);
    if (prog) prog->objv_w() = eval.LogitObjv();

    // py += .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
    if (V.weight.size()) {

      // tmp = (X.*X)*(V.*V)
      std::vector<real_t> vv = V.weight;
      for (auto& v : vv) v *= v;
      CHECK_EQ(vv.size(), V.pos.size() * V_dim);
      std::vector<real_t> xxvv(V.X.size * V_dim);
      SpMM::Times(V.XX, vv, &xxvv, nt);

      // V.XV = X*V
      V.XV.resize(xxvv.size());
      SpMM::Times(V.X, V.weight, &V.XV, nt);

      // py += .5 * sum((V.XV).^2 - xxvv)
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < py_.size(); ++i) {
        real_t* t = V.XV.data() + i * V_dim;
        real_t* tt = xxvv.data() + i * V_dim;
        real_t s = 0;
        for (int j = 0; j < V_dim; ++j) s += t[j] * t[j] - tt[j];
        py_[i] += .5 * s;
      }
    }

    // auc, acc, logloss, copc
    if (prog) {
      if (V.weight.size()) {
        prog->objv() = eval.LogitObjv();
      } else {
        prog->objv() = prog->objv_w();
      }
      prog->auc()    = eval.AUC();
      prog->new_ex() = w.X.size;
      prog->count()  = 1;
      prog->acc()    = eval.Accuracy(.5);
    }
  }

  /*!
   * \brief compute the gradients
   * p = - y ./ (1 + exp (y .* py));
   * grad_w = X' * p;
   * grad_u = X' * diag(p) * X * V  - diag((X.*X)'*p) * V
   */
  void CalcGrad(std::vector<real_t>* grad) {
    int nt = param_.nthreads;
    // p = ... (reuse py_)
    CHECK_EQ(py_.size(), w.X.size) << "call *evaluate* first";
#pragma omp parallel for num_threads(nt)
    for (size_t i = 0; i < py_.size(); ++i) {
      real_t y = w.X.label[i] > 0 ? 1 : -1;
      py_[i] = - y / ( 1 + exp ( y * py_[i] ));
    }

    // grad_w = ...
    SpMV::TransTimes(w.X, py_, &w.weight, nt);
    w.Save(grad);

    // grad_u = ...
    if (!V.weight.empty()) {
      int dim = param_.V_dim;

      // xxp = (X.*X)'*p
      size_t m = V.pos.size();
      std::vector<real_t> xxp(m);
      SpMM::TransTimes(V.XX, py_, &xxp, nt);

      // V = - diag(xxp) * V
      CHECK_EQ(V.weight.size(), dim * m);
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < m; ++i) {
        real_t* v = V.weight.data() + i * dim;
        for (int j = 0; j < dim; ++j) v[j] *= - xxp[i];
      }

      // V.XV = diag(p) * X * V
      size_t n = py_.size();
      CHECK_EQ(V.XV.size(), n * dim);
#pragma omp parallel for num_threads(nt)
      for (size_t i = 0; i < n; ++i) {
        real_t* y = V.XV.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[j] *= py_[i];
      }

      // V += X' * V.XV
      SpMM::TransTimes(V.X, V.XV, (real_t)1.0, V.weight, &V.weight, nt);

      // some preprocessing
      real_t gc = param_.V_grad_clipping;
      if (gc > 0) {
        for (real_t& g : V.weight) g = g > gc ? gc : ( g < -gc ? -gc : g);
      }

      real_t dp = param_.V_dropout;
      if (dp > 0) {
        for (real_t& g : V.weight) {
          if ((real_t)rand() / RAND_MAX > 1 - dp) g = 0;
        }
      }

      if (param_.V_grad_normalization) {
        real_t norm = 0;
        for (real_t g : V.weight) norm += g * g;
        if (norm < 1e-10) return;
        norm = sqrt(norm);
        for (real_t& g : V.weight) g = g / norm;
      }
      V.Save(grad);
    }
  }

  void Predict(std::vector<real_t>* pred) override {
    Evaluate(nullptr);
    *CHECK_NOTNULL(pred) = py_;
  }

 private:
  struct W {
    void Load(const dmlc::RowBlock<unsigned>& data,
              const std::vector<real_t>& model,
              const std::vector<int>& model_siz) {
      X = data;
      if (model_siz.empty()) {
        weight = model;
      } else {
        pos.resize(model_siz.size());
        weight.resize(model_siz.size());
        unsigned p = 0;
        for (size_t i = 0; i < model_siz.size(); ++i) {
          if (model_siz[i] == 0) {
            pos[i] = (unsigned)-1;
          } else {
            pos[i] = p; weight[i] = model[p]; p += model_siz[i];
          }
        }
        CHECK_EQ((size_t)p, model.size());
      }
    }

    void Save(std::vector<real_t>* grad) const {
      if (pos.empty()) {
        *CHECK_NOTNULL(grad) = weight;
      } else {
        for (int i = static_cast<int>(pos.size()); i > 0; --i) {
          if (pos[i-1] != static_cast<unsigned>(-1)) {
            size_t n = pos[i-1]+1;
            if (grad->size() < n) grad->resize(n);
            break;
          }
        }
        for (size_t i = 0; i < pos.size(); ++i) {
          if (pos[i] == static_cast<unsigned>(-1)) continue;
          grad->at(pos[i]) = weight[i];
        }
      }
    }

    void Clear() { weight.clear(); pos.clear(); }
    std::vector<real_t> weight;
    std::vector<unsigned> pos;
    dmlc::RowBlock<unsigned> X;
  } w;


  struct Embedding {
    void Load(int d,
              const dmlc::RowBlock<unsigned>& data,
              const std::vector<real_t>& model,
              const std::vector<int>& model_siz) {
      Clear();
      dim = d;
      if (dim == 0) return;
      std::vector<unsigned> col_map(model_siz.size());
      unsigned k = 0, p = 0;
      for (size_t i = 0; i < model_siz.size(); ++i) {
        if (model_siz[i] > 1) {
          CHECK_EQ(model_siz[i], dim + 1);
          pos.push_back(p+1);  // skip the first dim
          col_map[i] = ++ k;
        }
        p += model_siz[i];
      }
      CHECK_EQ((size_t)p, model.size());

      weight.resize(pos.size() * dim);
      for (size_t i = 0; i < pos.size(); ++i) {
        memcpy(weight.data()+i*dim, model.data()+pos[i], dim*sizeof(real_t));
      }

      // pick the columns of data with model_siz = dim + 1
      os_.push_back(0);
      for (size_t i = 0; i < data.size; ++i) {
        for (size_t j = data.offset[i]; j < data.offset[i+1]; ++j) {
          unsigned d = data.index[j];
          unsigned k = col_map[d];
          if (k > 0) {
            idx_.push_back(k-1);
            if (data.value) val_.push_back(data.value[j]);
          }
        }
        os_.push_back(idx_.size());
      }
      X.size = data.size;
      X.offset = BeginPtr(os_);
      X.value = BeginPtr(val_);
      X.index = BeginPtr(idx_);

      XX = X;
      if (X.value) {
        val2_.resize(X.offset[X.size]);
        for (size_t i = 0; i < val2_.size(); ++i) {
          val2_[i] = X.value[i] * X.value[i];
        }
        XX.value = BeginPtr(val2_);
      }
    }

    void Save(std::vector<real_t>* grad) const {
      if (dim == 0) return;
      CHECK_EQ(weight.size(), pos.size()*dim);
      size_t n = pos.back() + dim;
      if (grad->size() < n) grad->resize(n);
      for (size_t i = 0; i < pos.size(); ++i) {
        CHECK_LE(static_cast<size_t>(pos[i] + dim), grad->size());
        memcpy(grad->data()+pos[i], weight.data()+i*dim, dim*sizeof(real_t));
      }
    }

    void Clear() {
      weight.clear(); pos.clear(); XV.clear();
      val_.clear(); idx_.clear(); os_.clear(); val2_.clear();
    }
    std::vector<real_t> weight, XV;
    std::vector<unsigned> pos;
    dmlc::RowBlock<unsigned> X, XX;
    int dim;

   private:
    template <typename T>
    T* BeginPtr(std::vector<T>& vec) {
      return vec.empty() ? nullptr : vec.data();
    }
    std::vector<real_t> val_, val2_;
    std::vector<size_t> os_;
    std::vector<unsigned> idx_;
  } V;

  // predict_y
  std::vector<real_t> py_;
  // parameters
  FMParam param_;
};

}  // namespace difacto
#endif  // DIFACTO_LOSS_FM_H_
