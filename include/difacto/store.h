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
#include "dmlc/parameter.h"
#include "./sarray.h"
#include "./updater.h"
namespace difacto {

/**
 * \brief the store allows workers to get and set and model
 */
class Store {
 public:
  /**
   * \brief the factory function
   */
  static Store* Create();

  /** \brief default constructor */
  Store() { }
  /** \brief default deconstructor */
  virtual ~Store() { }

  static const int kFeaCount = 1;
  static const int kWeight = 2;
  static const int kGradient = 3;
  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /**
   * \brief push a list of (feature id, value) into the store
   *
   * @param sync_type
   * @param fea_ids
   * @param vals
   * @param lens
   * @param on_complete
   *
   * @return
   */
  virtual int Push(const SArray<feaid_t>& fea_ids,
                   int val_type,
                   const SArray<real_t>& vals,
                   const SArray<int>& lens,
                   const std::function<void()>& on_complete = nullptr) = 0;
  /**
   * \brief pull the values for a list of feature ids
   *
   * @param sync_type
   * @param fea_ids
   * @param vals
   * @param lens
   * @param on_complete
   *
   * @return
   */
  virtual int Pull(const SArray<feaid_t>& fea_ids,
                   int val_type,
                   SArray<real_t>* vals,
                   SArray<int>* lens,
                   const std::function<void()>& on_complete = nullptr) = 0;

  /**
   * \brief wait until a push or a pull is actually finished
   *
   * @param time
   */
  virtual void Wait(int time) = 0;

  /**
   * \brief return number of workers
   */
  virtual int NumWorkers() = 0;
  /**
   * \brief return number of servers
   */
  virtual int NumServers() = 0;
  /**
   * \brief return the rank of this node
   */
  virtual int Rank() = 0;

  /** \brief set an updater for the store, only required for a server node */
  void set_updater(const std::shared_ptr<Updater>& updater) {
    updater_ = updater;
  }
  /** \brief get the updater */
  std::shared_ptr<Updater> updater() { return updater_; }

 protected:
  std::shared_ptr<Updater> updater_;
};

}  // namespace difacto

#endif  // DIFACTO_STORE_H_
