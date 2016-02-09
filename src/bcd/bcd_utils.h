/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_BCD_BCD_UTILS_H_
#define DIFACTO_BCD_BCD_UTILS_H_
#include <mutex>
#include <condition_variable>
#include <vector>
#include <algorithm>
#include "dmlc/data.h"
#include "difacto/sarray.h"
namespace difacto {
namespace bcd {

class FeatureBlock {
 public:
  /**
   * \brief partition the whole feature space into blocks
   *
   * @param feagrp_nbits number of bit for encoding the feature group
   * @param feagrps a list of (feature_group, num_partitions_this_group)
   * @param feablks a list of feature blocks with the start and end ID
   */
  static void Partition(int feagrp_nbits,
                        const std::vector<std::pair<int,int>>& feagrps,
                        std::vector<Range>* feablks) {
    CHECK_EQ(feagrp_nbits % 4, 0) << "should be 0, 4, 8, ...";
    feablks->clear();
    for (auto f : feagrps) {
      int gid = f.first;
      Range rg(ReverseBytes(EncodeFeaGrpID(0, gid, feagrp_nbits)),
               ReverseBytes(EncodeFeaGrpID(std::numeric_limits<feaid_t>::max(), gid, feagrp_nbits)));
      for (int i = 0; i < f.second; ++i) {
        feablks->push_back(rg.Segment(i, f.second));
        CHECK(feablks->back().Valid());
      }
    }
    std::sort(feablks->begin(), feablks->end(),
              [](const Range& a, const Range& b) { return a.begin < b.begin;});
    for (size_t i = 1; i < feablks->size(); ++i) {
      auto& before = feablks->at(i-1), after = feablks->at(i);
      if (before.end < after.begin) ++before.end;
      CHECK_LE(before.end, after.begin);
    }
  }
};


/**
 * \brief count statistics for feature groups
 */
class FeaGroupStats {
 public:
  FeaGroupStats(int nbits) {
    CHECK_LE(nbits, 16);
    nbits_ = nbits;
    value_.resize((1<<nbits_)+2);
  }

  void Add(const dmlc::RowBlock<feaid_t>& rowblk) {
    real_t nrows = 0;
    for (size_t i = 0; i < rowblk.size; i+=skip_) {
      for (size_t j = rowblk.offset[i]; j < rowblk.offset[i+1]; ++j) {
        ++value_[DecodeFeaGrpID(rowblk.index[j], nbits_)];
      }
      ++nrows;
    }
    value_[1<<nbits_] += nrows;
    value_[(1<<nbits_)+1] += rowblk.size;
  }

  void Get(std::vector<real_t>* value) {
    *value = value_;
  }

 private:
  int nbits_;
  int skip_ = 10; // only count 10% data
  std::vector<real_t> value_;
};

/**
 * \brief monitor if or not a block is finished, thread safe
 */
class BlockTracker {
 public:
  BlockTracker(int num_blks) : done_(num_blks) { }
  /** \brief mark id as finished */
  void Finish(int id) {
    mu_.lock();
    done_[id] = 1;
    mu_.unlock();
    cond_.notify_all();
  }
  /** \brief block untill id is finished */
  void Wait(int id) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this, id] {return done_[id] == 1; });
  }
 private:
  std::mutex mu_;
  std::condition_variable cond_;
  std::vector<int> done_;
};

class Delta {
 public:
  /**
   * \brief init delta
   * @param delta
   */
  static void Init(size_t len, SArray<real_t>* delta, real_t init_val = 1.0) {
    delta->resize(len);
    for (size_t i = 0; i < len; ++i) (*delta)[i] = init_val;
  }

  /**
   * \brief update delta given the change of w
   */
  static void Update(real_t delta_w, real_t* delta, real_t max_val = 5.0) {
    *delta = std::min(max_val, static_cast<real_t>(std::abs(delta_w) * 2.0 + .1));
  }

};

}  // namespace bcd
}  // namespace difacto
#endif  // DIFACTO_BCD_BCD_UTILS_H_
