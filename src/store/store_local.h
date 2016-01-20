/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_STORE_STORE_LOCAL_H_
#define DIFACTO_STORE_STORE_LOCAL_H_
#include <string>
#include <vector>
#include <functional>
#include "difacto/store.h"
#include "difacto/updater.h"
#include "dmlc/parameter.h"
namespace difacto {


struct StoreLocalParam : public dmlc::Parameter<StoreLocalParam> {
  /** \brief type of model, default is sgd */
  std::string updater;
  DMLC_DECLARE_PARAMETER(StoreLocalParam) {
    DMLC_DECLARE_FIELD(updater).set_default("sgd");
  }
};

/**
 * \brief model sync within a machine
 */
class StoreLocal : public Store {
 public:
  StoreLocal() : updater_(nullptr) { }
  virtual ~StoreLocal() { delete updater_; }

  KWArgs Init(const KWArgs& kwargs) {
    auto remain = param_.InitAllowUnknown(kwargs);
    updater_ = Updater::Create(param_.updater);
    remain = updater_->Init(remain);
    return remain;
  }

  void Load(dmlc::Stream* fi, bool* has_aux) override {
    updater_->Load(fi, has_aux);
  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
    updater_->Save(save_aux, fo);
  }

  int Push(int sync_type,
           const SArray<feaid_t>& fea_ids,
           const SArray<real_t>& vals,
           const SArray<int>& lens,
           const std::function<void()>& on_complete) override {
    if (sync_type == kFeaCount) {
      updater_->AddCount(fea_ids, vals);
    } else {
      updater_->Update(fea_ids, vals, lens);
    }
    if (on_complete) on_complete();
    return time_++;
  }

  int Pull(int sync_type,
           const SArray<feaid_t>& fea_ids,
           SArray<real_t>* vals,
           SArray<int>* lens,
           const std::function<void()>& on_complete) override {
    updater_->Get(fea_ids, vals, lens);
    if (on_complete) on_complete();
    return time_++;
  }

  void Wait(int time) override { }

  int Rank() override { return 0; }
 private:
  int time_;
  Updater* updater_;
  StoreLocalParam param_;
};
}  // namespace difacto
#endif  // DIFACTO_STORE_STORE_LOCAL_H_
