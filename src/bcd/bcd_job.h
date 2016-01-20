#ifndef _BCD_JOB_H_
#define _BCD_JOB_H_
#include <string>
namespace difacto {
namespace bcd {

struct JobArgs {
  JobArgs() { }
  /** \brief construct from a string */
  JobArgs(const std::string& str) { ParseFromString(str); }

  void SerializeToString(std::string* string) const {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kIterateData = 3;
  static const int kPrepareData = 6;
  static const int kBuildFeatureMap = 7;
  /** \brief job type */
  int type;
  // /** \brief filename  */
  // std::string filename;
  // /** \brief number of partitions of this file */
  // int num_parts;
  // /** \brief the part will be processed */
  // int part_idx;
  /** \brief the order to process feature blocks */
  std::vector<int> fea_blks;

  /** \brief the ID range of each feature block */
  std::vector<Range> fea_blk_ranges;
};

struct PrepDataRets {

  void SerializeToString(std::string* string) const {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }

  std::vector<real_t> feagrp_avg;
};

struct IterDataRets {

  void SerializeToString(std::string* string) const {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }
  std::vector<real_t> progress;
};

}  // namespace bcd
}  // namespace difacto
#endif // _BCD_JOB_H_
