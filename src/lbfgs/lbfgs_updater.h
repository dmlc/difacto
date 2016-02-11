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
  float model_init_range;

  int tail_feature_filter;
  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;

  int m;
  DMLC_DECLARE_PARAMETER(LBFGSUpdaterParam) {
    DMLC_DECLARE_FIELD(tail_feature_filter).set_default(4);
    DMLC_DECLARE_FIELD(l1).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_default(.1);
    DMLC_DECLARE_FIELD(V_dim);
    DMLC_DECLARE_FIELD(m).set_default(10);
    DMLC_DECLARE_FIELD(model_init_range).set_default(.01);
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

    if (param_.V_dim) {
      model_lens_.resize(feaids_.size());
      size_t n = 0;
      for (size_t i = 0; i < feaids_.size(); ++i) {
        model_lens_[i] = 1 + (feacnts_[i] > param_.V_threshold ? param_.V_dim : 0);
        n +=  model_lens_[i];
      }
      models_.resize(n);

      n = 0;
      real_t scale = param_.model_init_range * 2;
      for (size_t i = 0; i < feaids_.size(); ++i) {
        for (int j = 1; j < model_lens_[i]; ++j) {
          models_[n+j] = (rand() / static_cast<real_t>(RAND_MAX) - .5) * scale;
        }
        n += model_lens_[i];
      }
    } else {
      models_.resize(feaids_.size());
    }
    return models_.size();
  }

  void PrepareCalcDirection(real_t alpha, std::vector<real_t>* aux) {
    if (y_.empty()) return;
    if (static_cast<int>(s_.size()) == param_.m) s_.erase(s_.begin());
    SArray<real_t> new_s(models_.size());
    lbfgs::Add(alpha, models_, &new_s);
    s_.push_back(new_s);
    twoloop_.CalcIncreB(s_, y_, grads_, aux);
  }

  /**
   * \brief return <∇f(w), p>
   *
   *
   * @return
   */
  real_t CalcDirection(const std::vector<real_t>& aux) {
    if (y_.size()) {
      twoloop_.ApplyIncreB(aux);
      twoloop_.CalcDirection(s_, y_, grads_, &models_);
    } else {
      memset(models_.data(), 0, models_.size()*sizeof(real_t));
      lbfgs::Add(-1, grads_, &models_);
    }
    // for (auto& p : models_) p = p > 5 ? 5 : (p < -5 ? -5 : p);
    return lbfgs::Inner(grads_, models_, nthreads_);
  }

  void Get(const SArray<feaid_t>& feaids,
           int value_type,
           SArray<real_t>* values,
           SArray<int>* lengths) override {
    if (value_type == Store::kFeaCount) {
      KVMatch(feaids_, feacnts_, feaids, values, ASSIGN, nthreads_);
    } else if (value_type == Store::kWeight) {
      feacnts_.clear();
      KVMatch(feaids_, models_, model_lens_, feaids, values, lengths,
              ASSIGN, nthreads_);
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
      // add y = new_grad - old_grad
      if (grads_.size()) {
        if (static_cast<int>(y_.size()) == param_.m) y_.erase(y_.begin());
        SArray<real_t> new_y;
        new_y.CopyFrom(values);
        lbfgs::Add(-1, grads_, &new_y);
        y_.push_back(new_y);
      }
      grads_ = values;
    } else {
      LOG(FATAL) << "...";
    }
  }

 private:
  LBFGSUpdaterParam param_;
  SArray<feaid_t> feaids_;
  SArray<real_t> feacnts_;

  std::vector<SArray<real_t>> s_, y_;

  /** \brief initilized with w, store p later */
  SArray<real_t> models_;
  SArray<int> model_lens_;
  SArray<real_t> grads_;

  lbfgs::Twoloop twoloop_;
  int nthreads_ = DEFAULT_NTHREADS;
};
}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_UPDATER_H_
