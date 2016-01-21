#ifndef DIFACTO_REPORTER_H_
#define DIFACTO_REPORTER_H_
#include <string>
#include <vector>
#include <functional>
#include "./base.h"
namespace difacto {
/**
 * \brief report to the scheduler
 */
class Reporter {
 public:
  /**
   * \brief factory function
   */
  static Reporter* Create();
  /** \brief constructor */
  Reporter() { }
  /** \brief deconstructor */
  virtual ~Reporter() { }
  /**
   * \brief init
   * @param kwargs keyword arguments
   * @return the unknown kwargs
   */
  virtual KWArgs Init(const KWArgs& kwargs) = 0;

  /////////////// functions for the scheduler node /////////////////
  /**
   * \brief the function to process the report sent by a node
   * @param node_id the node id
   * @param report the received report
   */
  typedef std::function<void(int node_id, const std::string& report)> Monitor;
  /**
   * \brief set the monitor function
   */
  virtual void SetMonitor(const Monitor& monitor) = 0;

  /////////////// functions for a server/worker node /////////////////

  /**
   * \brief report to the scheduler
   * \param report the report
   * \return the timestamp of the report
   */
  virtual int Report(const std::string& report) = 0;
  /**
   * \brief wait until a particular Report has been finished.
   * @param timestamp the timestamp of the report
   */
  virtual void Wait(int timestamp) = 0;
};

}  // namespace difacto
#endif  // DIFACTO_REPORTER_H_
