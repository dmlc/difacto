#ifndef DIFACTO_TRACKER_H_
#define DIFACTO_TRACKER_H_
namespace difacto {
#include <string>
#include <vector>
#include "./base.h"

/**
 * \brief a thread-safe distributed job tracker
 *
 * A job is a remote procedure call, which takes a string as the argments and
 * returns a string for the results. The tracker can be used by both producer
 * and consumer
 *
 * the producer can
 *
 * 1. continously add jobs into the tracker, which will send the jobs to the
 * coresponding consumers.
 * 2. set a monitor to process the results returned by the consumers.
 *
 * the consumer can set a function which will be called if a job is received from the
 * producer.
 */
class Tracker {
 public:
  /**
   * \brief factory function
   */
  static JobTracker* Create();
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

  /////////////// functions for the producer /////////////////
  /**
   * \brief add a job to the tracker
   *
   * Once added, the tracker will start to issue the job, no other call such as
   * Run is necessary
   *
   * \param node_id the node id this job will sent to, see \ref NodeID
   * \param args the job arguments
   */
  void Add(int node_id, std::string args) {
    Add({std::make_pair(node_id, args)});
  }
  /**
   * \brief add a list of jobs to the tracker
   * \param jobs the jobs to add
   */
  virtual void Add(const std::vector<std::par<int, std::string>>& jobs) = 0;
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
   * \brief the function to process the results returned by the consumers
   *
   * @param node_id the node id of the consumer
   * @param rets the returned results
   */
  typedef std::function<void(int node_id, const std::string& rets)> Monitor;
  /**
   * \brief set the monitor function
   */
  virtual void SetMonitor(const Monitor& monitor) = 0;

  /////////////// functions for the consumer /////////////////
  /**
   * \brief the definition of the consumer function
   * \param args the accepted job's arguments
   * \param rets the results send back to the producer
   */
  typedef std::function<void(const std::string& args, std::string* rets)> Consumer;
  /**
   * \brief set the consumer function
   */
  virtual void SetConsumer(const Consumer& consumer) = 0;
  /**
   * \brief block until the producer called \ref Stop
   */
  virtual void Wait() = 0;

};
}  // namespace difacto
#endif  // DIFACTO_TRACKER_H_
