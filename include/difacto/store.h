/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_STORE_H_
#define DIFACTO_STORE_H_
#include <memory>
#include <vector>
#include <string>
#include "./base.h"
#include "dmlc/io.h"
namespace difacto {


/**
 * \brief the store allows workers to get and set and model
 */
class Store {
 public:
  Store() { }
  virtual ~Store() { }

  const static int kFeaCount = 1;
  const static int kWeight = 2;
  const static int kGradient = 3;

  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief load the model
   * \param fi input stream
   * \param has_aux whether the loaded learner has aux data
   */
  virtual void Load(dmlc::Stream* fi, bool* has_aux) = 0;

  /**
   * \brief save the model
   * \param save_aux whether or not save aux data
   * \param fo output stream
   */
  virtual void Save(bool save_aux, dmlc::Stream *fo) const = 0;

  virtual int Push(int sync_type,
                   const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
                   const std::shared_ptr<std::vector<real_t>>& vals,
                   const std::shared_ptr<std::vector<int>>& lens,
                   const std::function<void()>& on_complete = nullptr) = 0;

  virtual int Pull(int sync_type,
                   const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
                   std::vector<real_t>* vals,
                   std::vector<int>* lens,
                   const std::function<void()>& on_complete = nullptr) = 0;

  virtual void Wait(int time) = 0;

  /**
   * \brief the factory function
   * \param type the type such as "local" or "dist"
   */
  static Store* Create(const std::string& type);
};

}  // namespace difacto

#endif  // DIFACTO_STORE_H_
