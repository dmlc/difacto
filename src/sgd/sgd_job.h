/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_JOB_H_
#define DIFACTO_SGD_SGD_JOB_H_
#include <string>
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
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed, -1 means all */
  int part_idx;
  /** \brief the current epoch */
  int epoch;

  Job() { }
  /** \brief construct from a string */
  explicit Job(const std::string& str) {
    ParseFromString(str);
  }
  void SerializeToString(std::string* str) const {
  }

  void ParseFromString(const std::string& str) {
  }
};

struct Progress {
  static std::string TextHead() {
    return " #ex new     |w|_0    |V|_0 logloss_w logloss accuracy AUC";
  }
  std::string TextString() {
    return "";
  }
  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(progress);
    delete ss;
  }
  void ParseFromString(const std::string& str) {
    auto pstr = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&pstr);
    ss->Read(&progress);
    delete ss;
  }

  void Merge(int node_id, const Progress& other) {
  }
  std::vector<real_t> progress;
};

}  // namespace sgd
}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_JOB_H_
