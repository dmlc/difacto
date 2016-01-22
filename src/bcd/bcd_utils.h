#ifndef _BCD_UTILS_H_
#define _BCD_UTILS_H_
#include "difacto/sarray.h"
namespace difacto {
namespace bcd {
/**
 * \brief partition the whole feature space into blocks
 *
 * @param feagrp_nbits number of bit for encoding the feature group
 * @param feagrps a list of (feature_group, num_partitions_this_group)
 * @param feablks a list of feature blocks with the start and end ID
 */
void PartitionFeatureSpace(int feagrp_nbits,
                           const std::vector<std::pair<int,int>>& feagrps,
                           std::vector<Range>* feablks) {
  CHECK_EQ(feagrp_nbits % 4, 0) << "should be 0, 4, 8, ...";
  feablks->clear();
  feaid_t key_max = std::numeric_limits<feaid_t>::max();
  for (auto f : feagrps) {
    CHECK_LT(1<<feagrp_nbits, f.first);
    Range g;
    g.begin = f.first;
    g.end = (key_max << feagrp_nbits) | f.first;
    for (int i = 0; i < f.second; ++i) {
      auto h = g.Segment(i, f.second);
      feablks->push_back(Range(ReverseBytes(h.begin), ReverseBytes(h.end)));
      CHECK(feablks->back().Valid());
    }
  }
  std::sort(feablks->begin(), feablks->end(),
            [](const Range& a, const Range& b) { return a.begin < b.begin;});
  for (size_t i = 1; i < feablks->size(); ++i) {
    CHECK_LT(feablks->at(i-1).end, feablks->at(i).begin);
    ++feablks->at(i-1).end;
  }
}

/**
 * \brief find the positionn of each feature block in the list of feature IDs
 *
 * @param feaids
 * @param feablks
 * @param positions
 */
void FindPosition(const SArray<feaid_t>& feaids,
                  const std::vector<Range>& feablks,
                  std::vector<Range>* positions) {
  size_t n = feablks.size();
  for (size_t i = 0; i < n; ++i) CHECK(feablks[i].Valid());
  for (size_t i = 1; i < n; ++i) CHECK_LE(feablks[i-1].end, feablks[i].begin);

  positions->resize(n);
  feaid_t const* begin = feaids.begin();
  feaid_t const* end = feaids.end();
  feaid_t const* cur = begin;

  for (size_t i = 0; i < n; ++i) {
    auto lb = std::lower_bound(cur, end, feablks[i].begin);
    auto ub = std::lower_bound(lb, end, feablks[i].begin);
    cur = ub;
    positions->at(i) = Range(lb - begin, ub - begin);
  }
}

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

/**
 * \brief count statistics for feature groups
 */
class FeaGroupStats {
 public:
  FeaGroupStats(int nbit) {
    CHECK_EQ(nbit % 4, 0) << "should be 0, 4, 8, ...";
    CHECK_LE(nbit, 16);
    nbit_ = nbit;
    occur_.resize(1<<nbit);
  }

  void Add(const dmlc::RowBlock<feaid_t>& rowblk) {
    for (size_t i = 0; i < rowblk.size; i+=skip_) {
      for (size_t j = rowblk.offset[i]; j < rowblk.offset[i+1]; ++j) {
        feaid_t f = rowblk.index[j];
        ++occur_[f-((f>>nbit_)<<nbit_)];
      }
      ++nrows_;
    }
  }

  /**
   * \brief get average nonzero features per row for each feature group
   */
  void Get(std::vector<real_t>* feagrp_avg) {
    feagrp_avg->resize(occur_.size());
    for (size_t i = 0; i < occur_.size(); ++i) {
      (*feagrp_avg)[i] = occur_[i] / nrows_;
    }
  }

 private:
  int nbit_;
  int skip_ = 10; // only count 10% data
  real_t nrows_ = 0;
  std::vector<real_t> occur_;
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

}  // namespace bcd
}  // namespace difacto
#endif  // _BCD_UTILS_H_
