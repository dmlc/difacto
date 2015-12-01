#pragma once
#include "ps/base.h"
namespace difacto {

using feaid_t = ps::Key;

/**
 * \brief the base class of a model updater
 * \tparam T the gradient/weight data type
 */
template <typename T>
class Model {
 public:
  /**
   * \brief load model
   * \param fi input stream
   */
  virtual void Load(dmlc::Stream* fi) = 0;

  /**
   * \brief save model
   * \param fo output stream
   */
  virtual void Save(dmlc::Stream *fo) const = 0;

  /**
   * \brief add feature count
   *
   * @param fea_ids the list of feature ids
   * @param fea_cnts the according counts
   */
  virtual void AddCount(const std::vector<feaid_t>& fea_ids,
                        const std::vector<int>& fea_cnts) = 0;

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
                   std::vector<T>* weights,
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
                      const std::vector<T>& gradients,
                      const std::vector<int>& gradient_lens) = 0;

  static Model<T>* Create(const Config& conf);
};

}  // namespace difacto
