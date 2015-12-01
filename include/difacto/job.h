/**
 * @file   job.h
 * @brief  job and job tracker
 */
#pragma once
#include <string>
#include <sstream>
#include "dmlc/io.h"
namespace difacto {

/**
 * \brief a job
 */
struct Job {
  /** \brief the job type  */
  enum Type {
    TRAINING,
    VALIDATION,
    PREDICTION,
    SAVE_MODEL,
    LOAD_MODEL
  } type;
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;

  /** \brief returns a readable string */
  std::string ShortDebugString() const {
    std::stringstream ss;
    ss << "epoch = " << epoch << ", "
       << (type == TRAINING ? "training," : "validation,")
       << filename << " " << part_idx << " / " << num_parts;
    return ss.str();
  }

  /** \brief load from stream */
  void Load(dmlc::Stream* fi) {
    fi->Read(&type, sizeof(type));
    fi->Read(&epoch, sizeof(epoch));
    fi->Read(&filename);
    fi->Read(&num_parts, sizeof(num_parts));
    fi->Read(&part_idx, sizeof(part_idx));
  }

  /** \brief save to stream */
  void Save(dmlc::Stream* fo) const {
    fo->Write(&type, sizeof(type));
    fo->Write(&epoch, sizeof(epoch));
    fo->Write(filename);
    fo->Write(&num_parts, sizeof(num_parts));
    fo->Write(&part_idx, sizeof(part_idx));
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
    consumer_ = consumer;
  }

  /**
   * \brief factory function
   * \param type can be "local" or "dist"
   */
  static JobTracker* Create(const std::string& type);

 protected:
  Consumer consumer_;
};


}  // namespace difacto
