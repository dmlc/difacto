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

/**
 * \brief model sync within a machine
 */
class StoreLocal : public Store {
 public:
  StoreLocal() { }
  virtual ~StoreLocal() { }

  KWArgs Init(const KWArgs& kwargs) { return kwargs; }

  int Push(const SArray<feaid_t>& fea_ids,
           int val_type,
           const SArray<real_t>& vals,
           const SArray<int>& lens,
           const std::function<void()>& on_complete) override {
    updater_->Update(fea_ids, val_type, vals, lens);
    if (on_complete) on_complete();
    return time_++;
  }

  int Pull(const SArray<feaid_t>& fea_ids,
           int val_type,
           SArray<real_t>* vals,
           SArray<int>* lens,
           const std::function<void()>& on_complete) override {
    updater_->Get(fea_ids, val_type, vals, lens);
    if (on_complete) on_complete();
    return time_++;
  }

  void Wait(int time) override { }

  int Rank() override { return 0; }
 private:
  int time_;
};
}  // namespace difacto
#endif  // DIFACTO_STORE_STORE_LOCAL_H_
