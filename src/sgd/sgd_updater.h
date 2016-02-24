/**
 * Copyright (c) 2015 by Contributors
 * @file   sgd.h
 * @brief  the stochastic gradient descent solver
 */
#ifndef DIFACTO_SGD_SGD_UPDATER_H_
#define DIFACTO_SGD_SGD_UPDATER_H_
#include <vector>
#include <limits>
#include "difacto/updater.h"
#include "./sgd_param.h"
#include "./sgd_utils.h"
#include "dmlc/io.h"
namespace difacto {
/**
 * \brief the weight entry for one feature
 */
struct SGDEntry {
 public:
  SGDEntry() { }
  ~SGDEntry() { delete [] V; }
  /** \brief the number of appearence of this feature in the data so far */
  real_t fea_cnt = 0;
  /** \brief w and its aux data */
  real_t w = 0, sqrt_g = 0, z = 0;
  /** \brief V and its aux data */
  real_t *V = nullptr;
};
/**
 * \brief sgd updater
 *
 * - w is updated by FTRL, which is a smooth version of adagrad works well with
 *   the l1 regularizer
 * - V is updated by adagrad
 */
class SGDUpdater : public Updater {
 public:
  SGDUpdater() {}
  virtual ~SGDUpdater() {}

  KWArgs Init(const KWArgs& kwargs) override;

  void Load(dmlc::Stream* fi, bool* has_aux) override {
       // TODO(mli)
  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
    // TODO(mli)
  }

  void Get(const SArray<feaid_t>& fea_ids,
           int value_type,
           SArray<real_t>* weights,
           SArray<int>* val_lens) override;


  void Update(const SArray<feaid_t>& fea_ids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& val_lens) override;

  void Evaluate(sgd::Progress* prog) const;

  const SGDUpdaterParam& param() const { return param_; }

 private:
  /** \brief update w by FTRL */
  void UpdateW(real_t gw, SGDEntry* e);

  /** \brief update V by adagrad */
  void UpdateV(real_t const* gV, SGDEntry* e);

  /** \brief init V */
  void InitV(SGDEntry* e);

  SGDUpdaterParam param_;
  std::unordered_map<feaid_t, SGDEntry> model_;
  bool has_aux_ = true;
};


}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_UPDATER_H_
