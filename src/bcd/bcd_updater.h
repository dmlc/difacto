#ifndef _BCD_UPDATER_H_
#define _BCD_UPDATER_H_
#include <string>
#include <vector>
#include <limits>
#include "difacto/updater.h"
#include "dmlc/parameter.h"
#include "dmlc/io.h"
#include "difacto/store.h"
#include "common/kv_match.h"
namespace difacto {

struct BCDUpdaterParam : public dmlc::Parameter<BCDUpdaterParam> {


};


class BCDModel {

};
class BCDUpdater : public Updater {
 public:
  BCDUpdater() { }
  virtual ~BCDUpdater() { }

  KWArgs Init(const KWArgs& kwargs) override {

  }

  void Load(dmlc::Stream* fi, bool* has_aux) override {

  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
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
    } else {
      LOG(FATAL) << "...";
    }
  }


  void Update(const SArray<feaid_t>& fea_ids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& offsets) override {
    if (value_type == Store::kFeaCount) {

    } else if (value_type == Store::kWeight) {

    } else {
      LOG(FATAL) << "...";
    }
  }
 private:

  void InitWeights() {

  }
  BCDUpdaterParam param_;

  SArray<feaid_t> feaids_;
  SArray<real_t> feacnt_;

  SArray<real_t> weights_;
  SArray<real_t> offsets_;
};



}  // namespace difacto


#endif  // _BCD_UPDATER_H_
