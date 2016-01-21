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
#include "dmlc/data.h"
#include "./sarray.h"
namespace difacto {
/**
 * \brief the basic class of a loss function
 */
class Loss {
 public:
  /**
   * \brief the factory function
   * \param type the loss type such as "fm"
   * \param num_threads number of threads
   */
  static Loss* Create(const std::string& type, int nthreads = DEFAULT_NTHREADS);
  /** \brief constructor */
  Loss() : nthreads_(DEFAULT_NTHREADS) { }
  /** \brief deconstructor */
  virtual ~Loss() { }
  /**
   * \brief init the loss function
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;
  /**
   * \brief predict given the data and model weights. often known as "forward"
   *
   * @param data the data
   * @param param model weights
   * @return pred the predict results
   */
  virtual void Predict(const dmlc::RowBlock<unsigned>& data,
                       const std::vector<SArray<char>>& param,
                       SArray<real_t>* pred) = 0;
  /**
   * \brief calculate gradient given the data and model weights. often known as "backward"
   * @param data the data
   * @param param model weights
   * @return grad the gradients
   */
  virtual void CalcGrad(const dmlc::RowBlock<unsigned>& data,
                        const std::vector<SArray<char>>& param,
                        SArray<real_t>* grad) = 0;
  /**
   * \brief set the number of threads
   */
  void set_nthreads(int nthreads) {
    CHECK_GT(nthreads, 1); CHECK_LT(nthreads, 50);
    nthreads_ = nthreads;
  }
 protected:
  int nthreads_;
};
}  // namespace difacto
#endif  // DIFACTO_LOSS_H_
