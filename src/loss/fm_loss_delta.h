#ifndef _FM_LOSS_DELTA_H_
#define _FM_LOSS_DELTA_H_
#include "difacto/loss.h"
#include "difacto/sarray.h"
namespace difacto {

/**
 * \brief the FM loss, different to \ref FMLoss, \ref FMLossDelta is feeded with
 * delta weight, and tranpose of X, each time
 */
class FMLossDelta : public Loss {
 public:
  /** \brief constructor */
  FMLossDelta() { }
  /** \brief deconstructor */
  virtual ~FMLossDelta() { }


  KWArgs Init(const KWArgs& kwargs) override {
  }

  /**
   * @param data X', the transpose of X
   * @param param parameters
   * - param[0], real_t, previous predict
   * - param[1], int, param[1][i] is the length of the gradient on the i-th feature
   *   and sum(param[2]) = length(grad)
   * @param grad output gradient
   */
  void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                const std::vector<SArray<char>>& param,
                SArray<real_t>* grad) override {

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

  }


};
}  // namespace difacto

#endif  // _FM_LOSS_DELTA_H_
