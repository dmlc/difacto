#ifndef _BCD_JOB_H_
#define _BCD_JOB_H_
#include <string>
#include "dmlc/memory_io.h"
namespace difacto {
namespace bcd {

/**
 * \brief y += x
 *
 * @param x
 * @param y
 */
template <typename Vec>
void Add(const Vec& x, Vec* y) {
  if (y->empty()) {
    *y = x;
  } else {
    CHECK_EQ(y->size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) (*y)[i] += x[i];
  }
}

struct JobArgs {
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

  JobArgs() { }
  /** \brief construct from a string */
  JobArgs(const std::string& str) {
    ParseFromString(str);
  }
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
};

struct PrepDataRets {
  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(value);
    delete ss;
  }
  void ParseFromString(const std::string& str) {
    auto pstr = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&pstr);
    ss->Read(&value);
    delete ss;
  }
  void Add(const PrepDataRets& other) {
    if (value.empty()) {
      value.resize(other.value.size());
    } else {
      CHECK_EQ(value.size(), other.value.size());
    }
    for (size_t i = 0; i < value.size(); ++i) {
      value[i] += other.value[i];
    }
  }
  std::vector<real_t> value;
};

/**
 * \brief the training progress
 *
 * value[0] : count
 * value[1] : objv
 * value[2] : auc
 * value[3] : acc
 * value[4] : ...
 */
struct Progress {
  void SerializeToString(std::string* str) const {
    dmlc::Stream* ss = new dmlc::MemoryStringStream(str);
    ss->Write(value);
    delete ss;
  }

  void ParseFromString(const std::string& str) {
    auto pstr = str;
    dmlc::Stream* ss = new dmlc::MemoryStringStream(&pstr);
    ss->Read(&value);
    delete ss;
  }

  void Add(int node_id, const Progress& other) {
    if (value.empty()) {
      value.resize(other.value.size());
    } else {
      CHECK_EQ(value.size(), other.value.size());
    }
    for (size_t i = 0; i < value.size(); ++i) {
      value[i] += other.value[i];
    }
  }


  std::vector<real_t> value;
};

}  // namespace bcd
}  // namespace difacto
#endif // _BCD_JOB_H_
