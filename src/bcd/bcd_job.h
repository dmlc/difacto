#ifndef _BCD_JOB_H_
#define _BCD_JOB_H_
#include <string>
namespace difacto {
namespace bcd {

struct JobArgs {
  /** \brief construct from a string */
  JobArgs(const std::string& str) { ParseFromString(str); }

  void SerializeToString(std::string* string) {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kTraining = 3;
  static const int kValidation = 4;
  static const int kPrediction = 5;
  static const int kPrepareTrainData = 6;
  static const int kPrepareValData = 7;
  /** \brief job type */
  int type;
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed */
  int part_idx;
  /** \brief the order to process feature blocks */
  std::vector<int> fea_blks;

  std::vector<std::pair<int, int>> fea_grps;
};

struct PrepDataRets {

  void SerializeToString(std::string* string) {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }

  std::vector<real_t> feablk_avg;
};

struct IterFeaBlkRets {

  void SerializeToString(std::string* string) {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }
};

}  // namespace bcd
}  // namespace difacto
#endif // _BCD_JOB_H_
