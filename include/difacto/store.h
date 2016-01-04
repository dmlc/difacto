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
namespace difacto {

struct StoreParam : public dmlc::Parameter<StoreParam> {
  /** \brief number of worker nodes */
  int num_workers;
  /** \brief number of server nodes */
  int num_servers;

  DMLC_DECLARE_PARAMETER(StoreParam) {
    DMLC_DECLARE_FIELD(num_workers);
    DMLC_DECLARE_FIELD(num_servers);
  }
};

/**
 * \brief the store allows workers to get and set and model
 */
class Store {
 public:
  Store() { }
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
  virtual KWArgs Init(const KWArgs& kwargs) {
    return param_.InitAllowUnknown(kwargs);
  }

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
  virtual int Push(int sync_type,
                   const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
                   const std::shared_ptr<std::vector<real_t>>& vals,
                   const std::shared_ptr<std::vector<int>>& lens,
                   const std::function<void()>& on_complete = nullptr) = 0;

  virtual int Push(int sync_type,
                   const SArray<feaid_t>& fea_ids,
                   const SArray<real_t>& vals,
                   const SArray<int>& lens,
                   const std::function<void()>& on_complete = nullptr)  {

  }
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
  virtual int Pull(int sync_type,
                   const std::shared_ptr<std::vector<feaid_t>>& fea_ids,
                   std::vector<real_t>* vals,
                   std::vector<int>* lens,
                   const std::function<void()>& on_complete = nullptr) = 0;

  virtual int Pull(int sync_type,
                   const SArray<feaid_t>& fea_ids,
                   SArray<real_t>* vals,
                   SArray<int>* lens,
                   const std::function<void()>& on_complete = nullptr) {

  }
  /**
   * \brief wait until a push or a pull is actually finished
   *
   * @param time
   */
  virtual void Wait(int time) = 0;

  /**
   * \brief return number of workers
   */
  int NumWorkers() { return param_.num_workers; }

  /**
   * \brief return number of servers
   */
  int NumServers() { return param_.num_servers; }

  /**
   * \brief return the rank of this node
   */
  virtual int Rank() = 0;

  /**
   * \brief the factory function
   */
  static Store* Create();

 private:
  StoreParam param_;
};

}  // namespace difacto

#endif  // DIFACTO_STORE_H_
