#pragma once
#include "./base.h"
#include "dmlc/data.h"
namespace difacto {

/**
 * \brief the basic class of a loss function
 */
class Loss {
 public:
  Loss() { }
  virtual ~Loss() { }

  /**
   * \brief init the loss function
   *
   * @param data X and Y
   * @param weights the weight entries
   * @param weight_lens the weight entry size. the i-th element is the length of
   * the i-th weight
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const RowBlock<unsigned>& data,
             const std::vector<real_t>& weights,
             const std::vector<int>& weight_lens,
             const KWArgs& kwargs) virtual = 0;

  /**
   * \brief evaluate the progress
   * \param prog the output progress
   */
  virtual void Evaluate(std::vector<real_t>* prog) = 0;

  /*!
   * \brief compute the gradients
   * \param grad the output gradients, often should have the same size as \ref model
   */
  virtual void CalcGrad(std::vector<real_t>* grad) = 0;

  /**
   * \brief predict
   * \param pred the prediction results, whose length should be equal to the
   * number of examples in \ref data
   */
  virtual void Predict(std::vector<real_t>* pred) = 0;

  /**
   * \brief the factory function
   * \param type the loss type such as "fm"
   */
  Loss* Create(const std::string& type);
};

}  // namespace difacto
