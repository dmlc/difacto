/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LBFGS_LBFGS_UPDATER_H_
#define DIFACTO_LBFGS_LBFGS_UPDATER_H_
#include <vector>
#include "./lbfgs_twoloop.h"
#include "difacto/updater.h"
namespace difacto {

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

  void InitWeight(std::vector<real_t>* rets) {
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
        unsigned seed = 0;
        real_t scale = param_.V_init_scale * 2;
        for (size_t i = 0; i < feaids_.size(); ++i) {
          for (int j = 1; j < weight_lens_[i]; ++j) {
            weights_[n+j] = (rand_r(&seed) / static_cast<real_t>(RAND_MAX) - .5) * scale;
          }
          n += weight_lens_[i];
        }
      }
    }

    rets->resize(2);
    (*rets)[0] = Evaluate();
    (*rets)[1] = weights_.size();
  }

  void Evaluate(lbfgs::Progress* prog) {
    prog->nnz_w = 0;
    for (auto w : weights_) if (w != 0) ++prog->nnz_w;
  }

  void PrepareCalcDirection(std::vector<real_t>* aux) {
    // add regularizer
    AddRegularizerGrad(&new_grads_);
    // it's epoch 0, no need to update s, y
    if (grads_.empty()) { grads_ = new_grads_; return; }
    // add new_grad - old_grad to y
    if (static_cast<int>(y_.size()) == param_.m) y_.erase(y_.begin());
    y_.resize(y_.size()+1);
    y_.back().CopyFrom(new_grads_);
    lbfgs::Add(-1, grads_, &y_.back(), nthreads_);
    grads_ = new_grads_;
    // update s
    lbfgs::Times(alpha_, &s_.back(), nthreads_);
    alpha_ = 0;
    twoloop_.CalcIncreB(s_, y_, grads_, aux);
  }

  /**
   * \brief return <∇f(w), p>
   * @return
   */
  real_t CalcDirection(const std::vector<real_t>& aux) {
    // calc direction
    SArray<real_t> dir;
    if (y_.size()) {
      twoloop_.ApplyIncreB(aux);
      twoloop_.CalcDirection(s_, y_, grads_, &dir);
    } else {
      dir.CopyFrom(grads_);
      lbfgs::Times(-1, &dir, nthreads_);
    }
    for (auto& p : dir) p = p > 5 ? 5 : (p < -5 ? -5 : p);
    // push into s_
    if (static_cast<int>(s_.size()) == param_.m) s_.erase(s_.begin());
    s_.push_back(dir);
    // return <p, g>
    return lbfgs::Inner(grads_, dir, nthreads_);
  }


  void LineSearch(real_t alpha, std::vector<real_t>* status) {
    lbfgs::Add(alpha - alpha_, s_.back(), &weights_, nthreads_);
    alpha_ = alpha;
    SArray<real_t> grads(weights_.size(), 0);
    AddRegularizerGrad(&grads);
    status->resize(2);
    (*status)[0] += Evaluate();
    (*status)[1] += lbfgs::Inner(grads, s_.back(), nthreads_);
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
      new_grads_ = values;
    } else {
      LOG(FATAL) << "...";
    }
  }

 private:
  void AddRegularizerGrad(SArray<real_t>* grads) {
    CHECK_EQ(grads->size(), weights_.size());
    if (weight_lens_.empty()) {
      lbfgs::Add(param_.l2, weights_, grads, nthreads_);
    } else {
      int n = 0;
      for (int l : weight_lens_) {
        for (int j = 0; j < l; ++j) {
          (*grads)[n] += (j == 0 ? param_.l2 : param_.V_l2) * weights_[n];
          ++n;
        }
      }
      CHECK_EQ(static_cast<size_t>(n), grads->size());
    }
  }

  /**
   * \brief return r(w)
   */
  real_t Evaluate() {
    real_t objv = 0;
    if (weight_lens_.empty()) {
      for (real_t w : weights_)  objv += .5 * param_.l2 * w * w;
    } else {
      int n = 0;
      for (int l : weight_lens_) {
        for (int j = 0; j < l; ++j) {
          real_t w = weights_[n++];
          objv += .5 * (j == 0 ? param_.l2 : param_.V_l2) * w * w;
        }
      }
      CHECK_EQ(static_cast<size_t>(n), weights_.size());
    }
    return objv;
  }

  LBFGSUpdaterParam param_;
  SArray<feaid_t> feaids_;
  SArray<real_t> feacnts_;

  std::vector<SArray<real_t>> s_, y_;

  SArray<real_t> weights_;
  SArray<int> weight_lens_;
  SArray<real_t> grads_, new_grads_;

  WeightInitializer weight_initializer_ = nullptr;
  lbfgs::Twoloop twoloop_;
  int nthreads_ = DEFAULT_NTHREADS;

  real_t alpha_ = 0;
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_UPDATER_H_
