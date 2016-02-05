#ifndef _LBFGS_UPDATER_H_
#define _LBFGS_UPDATER_H_
namespace difacto {

struct LBFGSUpdaterParam : public dmlc::Parameter<LBFGSUpdaterParam> {
  int V_dim = 0;
  int tail_feature_filter;
  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;
  DMLC_DECLARE_PARAMETER(LBFGSUpdaterParam) {
    DMLC_DECLARE_FIELD(tail_feature_filter).set_default(4);
    DMLC_DECLARE_FIELD(l1).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_default(.1);
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

  void Load(dmlc::Stream* fi, bool* has_aux) override {
  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
  }

  size_t InitWeights() {

  }

  void Get(const SArray<feaid_t>& feaids,
           int value_type,
           SArray<real_t>* values,
           SArray<int>* offsets) override {
    if (value_type == Store::kFeaCount) {
      values->resize(feaids.size());
      KVMatch(feaids_, feacnt_, feaids, values);
    } else if (value_type == Store::kWeight) {
      if (weights_.empty()) InitWeights();
      values->resize(feaids.size() * (param_.V_dim+1));
      if (param_.V_dim == 0) {
        KVMatch(feaids_, w_delta_, feaids, values);
      } else {
        offsets->resize(feaids.size());
        SArray<int> pos; FindPosition(feaids_, feaids, &pos);
        int *os = offsets->data(); os[0] = 0;
        real_t* val = values->data();
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          int start = offsets_[pos[i]+1];
          int len = offsets_[pos[i]+1] - start;
          os[1] = os[0] + len;
          memcpy(val, w_delta_.data() + start, len * sizeof(real_t));
          val += len; ++os;
        }
        values->resize(os[1]);
      }
    } else {
      LOG(FATAL) << "...";
    }
  }


  void Update(const SArray<feaid_t>& feaids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& offsets) override {
    if (value_type == Store::kFeaCount) {
      CHECK_EQ(values.size(), feaids.size());
      SArray<feaid_t> new_feaids;
      SArray<real_t> new_feacnt;
      KVUnion(feaids_, feacnt_, feaids, values, &new_feaids, &new_feacnt);
      feaids_ = new_feaids;
      feacnt_ = new_feacnt;
      // LL << DebugStr(feaids_);
      // LL << DebugStr(feacnt_);
    } else if (value_type == Store::kGradient) {
      if (weights_.empty()) InitWeights();
      SArray<int> pos; FindPosition(feaids_, feaids, &pos);
      if (offsets.empty()) {
        int k = 2;
        CHECK_EQ(values.size(), feaids.size()*k);
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          UpdateWeight(pos[i], values.data()+i*k, k);
        }
      } else {
        CHECK_EQ(offsets.size(), feaids.size());
        CHECK_EQ(offsets.back(), static_cast<int>(values.size()));
        for (size_t i = 0; i < pos.size(); ++i) {
          CHECK_NE(pos[i], -1);
          UpdateWeight(pos[i], values.data()+offsets[i], offsets[i+1]-offsets[i]);
        }
      }
    } else {
      LOG(FATAL) << "...";
    }
  }
 private:

  static void VectorFreeTwoLoop();
};
}  // namespace difacto
#endif  // _LBFGS_UPDATER_H_
