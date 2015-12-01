#pragma once
#include "./base.h"
#include "dmlc/io.h"
namespace difacto {

/**
 * \brief the base class of a model
 */
class Model {
 public:
  Model() { }
  virtual ~Model() { }

  /**
   * \brief init the model
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief load the model
   * \param fi input stream
   */
  virtual void Load(dmlc::Stream* fi) = 0;

  /**
   * \brief save the model
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
   * \param type the model type such as "fm"
   */
  static Model* Create(const std::string& type);
};

}  // namespace difacto
