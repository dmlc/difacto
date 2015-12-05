/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LEARNER_H_
#define DIFACTO_LEARNER_H_
#include <vector>
#include <string>
#include "./base.h"
#include "dmlc/io.h"
namespace difacto {
/**
 * \brief the base class of a learnerer.
 */
class Learner {
 public:
  Learner() { }
  virtual ~Learner() { }

  /**
   * \brief init the learner
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief load the learner
   * \param fi input stream
   * \param has_aux whether the loaded learner has aux data
   */
  virtual void Load(dmlc::Stream* fi, bool* has_aux) = 0;

  /**
   * \brief save the learner
   * \param save_aux whether or not save aux data
   * \param fo output stream
   */
  virtual void Save(bool save_aux, dmlc::Stream *fo) const = 0;

  /**
   * \brief add feature count
   *
   * @param fea_ids the list of feature ids
   * @param fea_cnts the according counts
   */
  virtual void AddCount(const std::vector<feaid_t>& fea_ids,
                        const std::vector<real_t>& fea_cnts) = 0;

  /**
   * \brief get the weights on the given features
   *
   * @param fea_ids the list of feature ids
   * @param weights the according weight on this feature ids, in format [w_0, V_0,
   * w_1, V_1, ...]
   * @param weight_lens the i-th element stores len([w_i, V_i]), could be empty
   * if there is only w
   */
  virtual void Get(const std::vector<feaid_t>& fea_ids,
                   std::vector<real_t>* weights,
                   std::vector<int>* weight_lens) = 0;

  /**
   * \brief update the weights given the gradients
   *
   * @param fea_ids the list of feature ids
   * @param gradients the according gradients on this feature ids, in format [gw_0, gV_0,
   * gw_1, gV_1, ...]
   * @param gradient_lens the i-th element stores len([gw_i, gV_i]), could be empty
   * if there is only w
   */
  virtual void Update(const std::vector<feaid_t>& fea_ids,
                      const std::vector<real_t>& grads,
                      const std::vector<int>& grad_lens) = 0;

  /**
   * \brief the factory function
   * \param type the learner type such as "fm"
   */
  static Learner* Create(const std::string& type);
};

}  // namespace difacto

#endif  // DIFACTO_LEARNER_H_
