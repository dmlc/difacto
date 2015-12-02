/**
 * @file   job.h
 * @brief  job and job tracker
 */
#pragma once
#include <string>
#include <sstream>
#include "dmlc/io.h"
#include "dmlc/parameter.h"
#include "./base.h"
namespace difacto {

/**
 * \brief a job
 */
struct Job : public dmlc::Parameter<Job> {
  static const int kLoadModel = 0;
  static const int kSaveModel = 1;
  static const int kTraining = 2;
  static const int kValidation = 3;
  static const int kPrediction = 4;
  /** \brief the job type  */
  int type;
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;

  DMLC_DECLARE_PARAMETER(Job) {
    DMLC_DECLARE_FIELD(num_parts).set_range(0, 100000).set_default(0);
    DMLC_DECLARE_FIELD(part_idx).set_range(0,  100000).set_default(0);
    DMLC_DECLARE_FIELD(epoch).set_range(0, 10000).set_default(0);
    DMLC_DECLARE_FIELD(filename);
    DMLC_DECLARE_FIELD(type).set_range(0, 5);
  }
};

/**
 * \brief a thread-safe distributed job tracker
 *
 * - the producer can continously add jobs into the tracker
 * - the tracker will assign the jobs to the consumers
 */
class JobTracker {
 public:
  JobTracker() { }
  virtual ~JobTracker() { }
  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;
  /**
   * \brief add a list of jobs to the tracker
   * \param jobs the jobs to add
   */
  virtual void Add(const std::vector<Job>& jobs) = 0;
  /**
   * \brief return the number of unfinished job
   */
  virtual int NumRemains() = 0;
  /**
   * \brief clear all unfinished jobs
   *
   * stop to assign new jobs to consumers. however, it dose nothing for jobs
   * that are running now.
   */
  virtual void Clear() = 0;
  /**
   * \brief stop the tracker
   *
   * it first wait all jobs are done, and then issue a STOP job to all consumers
   * and wait they are done
   */
  virtual void Stop() = 0;

  /**
   * \brief the consumer function
   */
  typedef std::function<void(const Job&)> Consumer;
  /**
   * \brief set the consumer function
   */
  void SetConsumer(const Consumer& consumer) {
    consumer_ =  consumer;
  }
  /**
   * \brief block until the producer called \ref Stop
   */
  virtual void Wait() = 0;

  /**
   * \brief factory function
   * \param type can be "local" or "dist"
   */
  static JobTracker* Create(const std::string& type);

 protected:
  Consumer consumer_;
};


}  // namespace difacto
