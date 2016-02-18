/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_JOB_H_
#define DIFACTO_SGD_SGD_JOB_H_
#include <string>
#include <vector>
#include "dmlc/memory_io.h"
namespace difacto {
namespace sgd {

/**
 * \brief a sgd job
 */
struct Job {
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kTraining = 3;
  static const int kValidation = 4;
  int type;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;
  Job() { }
  void SerializeToString(std::string* str) const {
    *str = std::string(reinterpret_cast<char const*>(this), sizeof(Job));
  }

  void ParseFromString(const std::string& str) {
    CHECK_EQ(str.size(), sizeof(Job));
    memcpy(this, str.data(), sizeof(Job));
  }
};

struct Progress {
  real_t objv = 0;  // objective value on training data
  real_t auc = 0;   // auc
  real_t nnz_w = 0; // |w|_0
  real_t w_size = 0;  // size of w
  real_t nrows = 0;   // number of examples

  std::string TextString() {
    return std::to_string(auc);
  }

  void SerializeToString(std::string* str) const {
    *str = std::string(reinterpret_cast<char const*>(this), sizeof(Progress));
  }
  void ParseFromString(const std::string& str) {
    if (str.empty()) return;
    CHECK_EQ(str.size(), sizeof(Progress));
    memcpy(this, str.data(), sizeof(Progress));
  }

  void Merge(const std::string& str) {
  }

  void Merge(const Progress& other) {
    size_t n = sizeof(Progress) / sizeof(real_t);
    auto a = reinterpret_cast<real_t*>(this);
    auto b = reinterpret_cast<real_t const*>(&other);
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
  }
};

}  // namespace sgd
}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_JOB_H_
