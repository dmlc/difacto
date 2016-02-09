/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LOSS_FM_LOSS_DELTA_H_
#define DIFACTO_LOSS_FM_LOSS_DELTA_H_
#include <vector>
#include "difacto/sarray.h"
#include "./fm_loss.h"
namespace difacto {

/**
 * \brief the FM loss, different to \ref FMLoss, \ref FMLossDelta is feeded with
 * delta weight, and tranpose of X, each time
 */
class FMLossDelta : public FMLoss {
 public:
  /** \brief constructor */
  FMLossDelta() { }
  /** \brief deconstructor */
  virtual ~FMLossDelta() { }

  KWArgs Init(const KWArgs& kwargs) override {
    return kwargs;
  }

  /**
   * @param data X', the transpose of X
   * @param param parameters
   * - param[0], real_t, previous predict
   * - param[1], int, param[1][i] is the length of the gradient on the i-th feature
   *   and sum(param[2]) = length(grad)
   * - param[2], real_t, the prediction (results of \ref Predict)
   * @param grad output gradient
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {
    // TODO(mli)
  }

  /**
   * @param data X', the transpose of X
   * @param param parameters
   * - param[0], real_t, previous predict
   * - param[1], real_t, delta weight, namely new_w - old_w
   * - param[2], int, param[2][i] is the length of delta_w[i].
   *   and sum(param[2]) = length(delta_w)
   * @param pred output prediction, it may overwrite param[0]
   */
  void Predict(const dmlc::RowBlock<unsigned>& data,
               const std::vector<SArray<char>>& param,
               SArray<real_t>* pred) override {
    *CHECK_NOTNULL(pred) = param[0];
    FMLoss::Predict(data, {param[1], param[2]}, pred);
  }
};
}  // namespace difacto

#endif  // DIFACTO_LOSS_FM_LOSS_DELTA_H_
