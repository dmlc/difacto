#ifndef DIFACTO_PROGRESS_H_
#define DIFACTO_PROGRESS_H_
#include <vector>
#include <string>
#include "./base.h"
namespace difacto {
/**
 * \brief the training progress
 */
struct Progress {
 public:
  Progress() : data(8) { }

  /** \brief accessors & mutators */
  real_t& objv() { return data[0]; }
  real_t objv() const { return data[0]; }

  real_t& auc() { return data[1]; }
  real_t auc() const { return data[1]; }

  real_t& acc() { return data[3]; }
  real_t acc() const { return data[3]; }

  real_t& objv_w() { return data[2]; }
  real_t objv_w() const { return data[2]; }

  real_t& count() { return data[4]; }
  real_t count() const { return data[4]; }

  real_t& new_ex() { return data[5]; }
  real_t new_ex() const { return data[5]; }

  real_t& new_w() { return data[6]; }
  real_t new_w() const { return data[6]; }

  real_t& new_V() { return data[7]; }
  real_t new_V() const { return data[7]; }

  /**
   * \brief merge from another progress
   */
  void Merge(const Progress& other) {
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] += other.data[i];
    }
  }

  /**
   * \brief the actual data
   */
  std::vector<real_t> data;
};

/**
 * \brief returns a readable string for the progress
 */
class ProgressPrinter {
 public:
  /**
   * \brief return a readable header string
   */
  std::string Head() const {
    return " #ex new     |w|_0    |V|_0 logloss_w logloss accuracy AUC";
  }

  /**
   * \brief return a readable string
   * \param show_V whether or not show V
   */
  std::string Body(const Progress& cur);

 private:
  Progress prev_;
};

/**
 * \brief collects progress from all nodes. This class is thread safe.
 */
class ProgressMonitor {
 public:
  virtual ~ProgressMonitor() { }

  /**
   * \brief add progress to the monitor
   */
  virtual void Add(const Progress& prog) = 0;

  /**
   * \brief get the progress from the monitor
   *
   */
  virtual void Get(Progress* prog) = 0;

  /**
   * \brief the factory function
   */
  static ProgressMonitor* Create();
};

}  // namespace difacto
#endif /* DIFACTO_PROGRESS_H_ */
