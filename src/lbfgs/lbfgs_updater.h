/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LBFGS_LBFGS_UPDATER_H_
#define DIFACTO_LBFGS_LBFGS_UPDATER_H_
#include <vector>
#include "./lbfgs_twoloop.h"
#include "difacto/updater.h"
namespace difacto {

struct LBFGSUpdaterParam : public dmlc::Parameter<LBFGSUpdaterParam> {
  int V_dim;

  /** \brief features with occurence < threshold have no embedding */
  int V_threshold = 2;

  /** \brief initialize the weights into [-x, +x] */
  float weight_init_range;

  int tail_feature_filter;
  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;

  float V_l2;

  int m;
  DMLC_DECLARE_PARAMETER(LBFGSUpdaterParam) {
    DMLC_DECLARE_FIELD(tail_feature_filter).set_default(4);
    DMLC_DECLARE_FIELD(l1).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_default(.1);
    DMLC_DECLARE_FIELD(V_l2).set_default(.01);
    DMLC_DECLARE_FIELD(V_dim);
    DMLC_DECLARE_FIELD(V_threshold).set_default(0);
    DMLC_DECLARE_FIELD(m).set_default(10);
    DMLC_DECLARE_FIELD(weight_init_range).set_default(.01);
  }
};

class LBFGSUpdater : public Updater {
 public:
  LBFGSUpdater() { }
  virtual ~LBFGSUpdater() { }

  KWArgs Init(const KWArgs& kwargs) override {
    return param_.InitAllowUnknown(kwargs);
  }

  const LBFGSUpdaterParam& param() const { return param_; }

  void Load(dmlc::Stream* fi, bool* has_aux) override { }

  void Save(bool save_aux, dmlc::Stream *fo) const override { }

  typedef std::function<void(
      const SArray<int>& weight_lens, SArray<real_t>* weights)> WeightInitializer;

  void SetWeightInitializer(const WeightInitializer& initer) {
    weight_initializer_ = initer;
  }

  size_t InitWeights() {
    if (param_.tail_feature_filter > 0) {
      SArray<feaid_t> filtered_ids;
      SArray<real_t> filtered_cnts;
      lbfgs::RemoveTailFeatures(
          feaids_, feacnts_, param_.tail_feature_filter, &filtered_ids);
      KVMatch(feaids_, feacnts_, filtered_ids, &filtered_cnts, ASSIGN, nthreads_);
      feaids_ = filtered_ids;
      feacnts_ = filtered_cnts;
    }

    size_t n = 0;
    if (param_.V_dim) {
      weight_lens_.resize(feaids_.size());
      for (size_t i = 0; i < feaids_.size(); ++i) {
        weight_lens_[i] = 1 + (feacnts_[i] > param_.V_threshold ? param_.V_dim : 0);
        n +=  weight_lens_[i];
      }
    } else {
      n = feaids_.size();
    }
    weights_.resize(n);

    if (weight_initializer_) {
      weight_initializer_(weight_lens_, &weights_);
    } else {
      if (param_.V_dim) {
        n = 0;
        real_t scale = param_.weight_init_range * 2;
        for (size_t i = 0; i < feaids_.size(); ++i) {
          for (int j = 1; j < weight_lens_[i]; ++j) {
            weights_[n+j] = (rand() / static_cast<real_t>(RAND_MAX) - .5) * scale;
          }
          n += weight_lens_[i];
        }
      }
    }

    return weights_.size();
  }

  void PrepareCalcDirection(real_t alpha, std::vector<real_t>* aux) {
    if (y_.empty()) return;
    auto& p = s_.back();
    lbfgs::Add(alpha - 1.0, p, &p);
    lbfgs::Add(1.0, p, &weights_);
    twoloop_.CalcIncreB(s_, y_, grads_, aux);
  }

  /**
   * \brief return <∇f(w), p>
   *
   *
   * @return
   */
  real_t CalcDirection(const std::vector<real_t>& aux) {
    // calc direction
    SArray<real_t> dir;
    if (y_.size()) {
      twoloop_.ApplyIncreB(aux);
      twoloop_.CalcDirection(s_, y_, grads_, &dir);
    } else {
      dir.resize(grads_.size(), 0);
      lbfgs::Add(-1, grads_, &dir);
    }
    for (auto& p : dir) p = p > 5 ? 5 : (p < -5 ? -5 : p);

    // push into s_
    if (static_cast<int>(s_.size()) == param_.m) s_.erase(s_.begin());
    s_.push_back(dir);

    return lbfgs::Inner(grads_, dir, nthreads_);
  }

  void Get(const SArray<feaid_t>& feaids,
           int value_type,
           SArray<real_t>* values,
           SArray<int>* lengths) override {
    if (value_type == Store::kFeaCount) {
      KVMatch(feaids_, feacnts_, feaids, values, ASSIGN, nthreads_);
    } else if (value_type == Store::kWeight) {
      feacnts_.clear();
      if (s_.size()) {
        KVMatch(feaids_, s_.back(), weight_lens_, feaids, values, lengths,
                ASSIGN, nthreads_);
      } else {
        KVMatch(feaids_, weights_, weight_lens_, feaids, values, lengths,
                ASSIGN, nthreads_);
      }
    } else {
      LOG(FATAL) << "...";
    }
  }

  void Update(const SArray<feaid_t>& feaids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& lengths) override {
    if (value_type == Store::kFeaCount) {
      feaids_ = feaids; feacnts_ = values;
    } else if (value_type == Store::kGradient) {
      CHECK_EQ(feaids_.size(), feaids.size());
      // add l2 regularizer
      auto new_grad = values;
      AddRegularizer(lengths, weights_, &new_grad);
      real_t v = 0; for (auto g : new_grad) v += g*g; LL << v;
      // add y = new_grad - old_grad
      if (grads_.size()) {
        if (static_cast<int>(y_.size()) == param_.m) y_.erase(y_.begin());
        SArray<real_t> new_y;
        new_y.CopyFrom(new_grad);
        lbfgs::Add(-1, grads_, &new_y);
        y_.push_back(new_y);
      }
      grads_ = new_grad;
    } else {
      LOG(FATAL) << "...";
    }
  }

 private:
  // g += l2 * w
  void AddRegularizer(const SArray<int>& lens,
                      const SArray<real_t>& weights,
                      SArray<real_t>* grads) {
    CHECK_EQ(grads->size(), weights.size());
    int n = 0;
    for (int l : lens) {
      for (int j = 0; j < l; ++j) {
        (*grads)[n] += (j == 0 ? param_.l2 : param_.V_l2) * weights[n];
        ++n;
      }
    }
    CHECK_EQ(static_cast<size_t>(n), grads->size());
  }

  LBFGSUpdaterParam param_;
  SArray<feaid_t> feaids_;
  SArray<real_t> feacnts_;

  std::vector<SArray<real_t>> s_, y_;

  SArray<real_t> weights_;
  SArray<int> weight_lens_;
  SArray<real_t> grads_;

  WeightInitializer weight_initializer_ = nullptr;
  lbfgs::Twoloop twoloop_;
  int nthreads_ = DEFAULT_NTHREADS;
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_UPDATER_H_
