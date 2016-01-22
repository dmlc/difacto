#ifndef _BCD_UPDATER_H_
#define _BCD_UPDATER_H_
#include <string>
#include <vector>
#include <limits>
#include "difacto/updater.h"
#include "dmlc/parameter.h"
#include "dmlc/io.h"
namespace difacto {

struct BCDUpdaterParam : public dmlc::Parameter<BCDUpdaterParam> {


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

  void Get(const SArray<feaid_t>& fea_ids,
           int value_type,
           SArray<real_t>* weights,
           SArray<int>* offsets) override {}


  void Update(const SArray<feaid_t>& fea_ids,
              int value_type,
              const SArray<real_t>& values,
              const SArray<int>& offsets) override {}
 private:
  BCDUpdaterParam param_;


};



}  // namespace difacto


#endif  // _BCD_UPDATER_H_
