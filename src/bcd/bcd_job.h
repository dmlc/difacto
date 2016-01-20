#ifndef _BCD_JOB_H_
#define _BCD_JOB_H_
#include <string>
#include "dmlc/memory_io.h"
namespace difacto {
namespace bcd {

struct JobArgs {
  JobArgs() { }
  /** \brief construct from a string */
  JobArgs(const std::string& str) { ParseFromString(str); }

  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(type);
    ss->Write(fea_blks);
    ss->Write(fea_blk_ranges.size());
    for (auto r : fea_blk_ranges) {
      ss->Write(r.begin); ss->Write(r.end);
    }
    delete ss;
  }

  void ParseFromString(const std::string& str) {
    auto pstr = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&pstr);
    ss->Read(&type);
    ss->Read(&fea_blks);
    size_t size; ss->Read(&size);
    fea_blk_ranges.resize(size);
    for (size_t i = 0; i < size; ++i) {
      ss->Read(&fea_blk_ranges[i].begin);
      ss->Read(&fea_blk_ranges[i].end);
    }
    delete ss;
  }

  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kIterateData = 3;
  static const int kPrepareData = 6;
  static const int kBuildFeatureMap = 7;
  /** \brief job type */
  int type;
  /** \brief the order to process feature blocks */
  std::vector<int> fea_blks;
  /** \brief the ID range of each feature block */
  std::vector<Range> fea_blk_ranges;
};

struct PrepDataRets {
  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(feagrp_avg);
    delete ss;
  }
  void ParseFromString(const std::string& str) {
    auto pstr = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&pstr);
    ss->Read(&feagrp_avg);
    delete ss;
  }

  std::vector<real_t> feagrp_avg;
};

struct IterDataRets {
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
  std::vector<real_t> progress;
};

}  // namespace bcd
}  // namespace difacto
#endif // _BCD_JOB_H_
