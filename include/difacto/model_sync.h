/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_MODEL_SYNC_H_
#define DIFACTO_MODEL_SYNC_H_
#include <memory>
#include <vector>
#include <string>
#include "./base.h"
namespace difacto {

/**
 * \brief the commnunication interface of a model, which is used by
 * workers to get / set the model
 */
class ModelSync {
 public:
  ModelSync() { }
  virtual ~ModelSync() { }

  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief use shared pointer for communication
   */
  template <typename V>
  using SharedVector = std::shared_ptr<std::vector<V>>;

  /** \brief the callback function type */
  typedef std::function<void()> Callback;

  virtual int Push(const SharedVector<feaid_t>& fea_ids,
                   const SharedVector<real_t>& vals,
                   const SharedVector<int>& lens,
                   const Callback& on_complete = Callback()) = 0;

  virtual int Pull(const SharedVector<feaid_t>& fea_ids,
                   std::vector<real_t>* vals,
                   std::vector<int>* lens,
                   const Callback& on_complete = Callback()) = 0;


  virtual void Wait(int time) = 0;

  /**
   * \brief the factory function
   * \param type the type such as "local" or "dist"
   */
  static ModelSync* Create(const std::string& type);
};

}  // namespace difacto

#endif  // DIFACTO_MODEL_SYNC_H_
