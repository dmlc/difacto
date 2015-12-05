#ifndef DIFACTO_STORE_STORE_LOCAL_H_
#define DIFACTO_STORE_STORE_LOCAL_H_
#include "difacto/store.h"
#include "difacto/learner.h"
#include "dmlc/parameter.h"
namespace difacto {


struct StoreLocalParam : public dmlc::Parameter<StoreLocalParam> {
  /** \brief type of model, default is sgd */
  std::string learner;
  DMLC_DECLARE_PARAMETER(StoreLocalParam) {
    DMLC_DECLARE_FIELD(learner).set_default("sgd");
  }
};

/**
 * \brief model sync within a machine
 */
class StoreLocal : public Store {
 public:
  StoreLocal() : learner_(nullptr) { }
  virtual ~StoreLocal() { delete learner_; }

  KWArgs Init(const KWArgs& kwargs) {
    auto remain = param_.InitAllowUnknown(kwargs);
    learner_ = Learner::Create(param_.learner);
    return remain;
  }

  void Load(dmlc::Stream* fi, bool* has_aux) override {
    learner_->Load(fi, has_aux);
  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
    learner_->Save(save_aux, fo);
  }

  int Push(int sync_type,
           const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
           const std::shared_ptr<std::vector<real_t>>& vals,
           const std::shared_ptr<std::vector<int>>& lens,
           const std::function<void()>& on_complete) override {
    if (sync_type == kFeaCount) {
      learner_->AddCount(*fea_ids, *vals);
    } else {
      learner_->Update(*fea_ids, *vals, *lens);
    }
    if (on_complete) on_complete();
    return time_++;
  }

  int Pull(int sync_type,
           const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
           std::vector<real_t>* vals,
           std::vector<int>* lens,
           const std::function<void()>& on_complete) override {
    learner_->Get(*fea_ids, vals, lens);
    if (on_complete) on_complete();
    return time_++;
  }

  void Wait(int time) override { }

 private:
  int time_;
  Learner* learner_;
  StoreLocalParam param_;
};
}  // namespace difacto
#endif  // DIFACTO_STORE_STORE_LOCAL_H_
