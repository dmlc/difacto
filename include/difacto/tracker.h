/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_TRACKER_H_
#define DIFACTO_TRACKER_H_
#include <string>
#include <vector>
#include <functional>
#include "./base.h"
namespace difacto {
/**
 * \brief a thread-safe distributed job tracker
 *
 * A job is a remote procedure call, which takes a string as the argments and
 * returns a string for the results. The tracker can be used by both producer
 * and executor
 *
 * the producer can
 *
 * 1. continously add jobs into the tracker, which will send the jobs to the
 * coresponding executors.
 * 2. set a monitor to process the results returned by the executors.
 *
 * the executor can set a function which will be called if a job is received from the
 * producer.
 */
class Tracker {
 public:
  /**
   * \brief factory function
   */
  static Tracker* Create();
  /** \brief constructor */
  Tracker() { }
  /** \brief deconstructor */
  virtual ~Tracker() { }
  /**
   * \brief init
   *
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /////////////// functions for the scheduler node /////////////////
  /**
   * \brief issue a job to the executor
   *
   * If the node ID of the executor is a group, e.g. kWorkerGroup, then this job
   * could be sent to any node in that group. Given one executor, the tracker
   * sends a job only if the previous one has been finished.
   *
   * This function returns immediately once the job is queued. Use
   * \ref NumRemains to query if or not the job is finished.
   *
   * \param node_id the node id of the executor, can be a group id
   * \param args the job arguments
   */
  void Issue(int node_id, std::string args) {
    Issue({std::make_pair(node_id, args)});
  }
  /**
   * \brief issue a list of jobs to the executors
   * \param jobs the jobs to add
   */
  virtual void Issue(const std::vector<std::pair<int, std::string>>& jobs) = 0;

  /**
   * \brief return the number of unfinished job
   */
  virtual int NumRemains() = 0;
  /**
   * \brief clear all unfinished jobs
   *
   * stop to assign new jobs to executors. however, it dose nothing for jobs
   * that are running now.
   */
  virtual void Clear() = 0;
  /**
   * \brief stop the tracker
   *
   * it first wait all jobs are done, and then issue a STOP signals to all executors
   * and wait they are done
   */
  virtual void Stop() = 0;
  /**
   * \brief the function to process the results returned by the executors
   *
   * @param node_id the node id of the executor
   * @param rets the returned results
   */
  typedef std::function<void(int node_id, const std::string& rets)> Monitor;
  /**
   * \brief set the monitor function
   */
  virtual void SetMonitor(const Monitor& monitor) = 0;

  /////////////// functions for a server/worker node /////////////////
  /**
   * \brief the definition of the executor function
   * \param args the accepted job's arguments
   * \param rets the results send back to the producer
   */
  typedef std::function<void(const std::string& args, std::string* rets)> Executor;
  /**
   * \brief set the executor function
   */
  virtual void SetExecutor(const Executor& executor) = 0;
  /**
   * \brief block until the producer called \ref Stop
   */
  virtual void Wait() = 0;

};
}  // namespace difacto
#endif  // DIFACTO_TRACKER_H_
