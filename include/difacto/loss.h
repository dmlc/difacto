/**
 * Copyright (c) 2015 by Contributors
 * @file   loss.h
 * @brief  the basic class of a loss function
 */
#ifndef DIFACTO_LOSS_H_
#define DIFACTO_LOSS_H_
#include <string>
#include <vector>
#include "./base.h"
#include "./progress.h"
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
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief init the loss with data
   *
   * @param data X and Y
   * @param weights the weight entries
   * @param weight_lens the weight entry size. the i-th element is the length of
   * the i-th weight. could be empty
   */
  virtual void InitData(const dmlc::RowBlock<unsigned>& data,
                        const std::vector<real_t>& weights,
                        const std::vector<int>& weight_lens) = 0;
  /**
   * \brief evaluate the progress
   * \param prog the output progress
   */
  virtual void Evaluate(Progress* prog) = 0;

  /*!
   * \brief compute the gradients
   * \param grad the output gradients. grad could be empty.
   */
  virtual void CalcGrad(std::vector<real_t>* grad) = 0;

  /**
   * \brief predict
   * \param pred the prediction results, whose length should be equal to the
   * number of examples in \ref data
   */
  virtual void Predict(std::vector<real_t>* pred) = 0;

  /**
   * \brief Clear data
   */
  virtual void Clear() = 0;

  /**
   * \brief the factory function
   * \param type the loss type such as "fm"
   */
  static Loss* Create(const std::string& type);
};

}  // namespace difacto

#endif  // DIFACTO_LOSS_H_
