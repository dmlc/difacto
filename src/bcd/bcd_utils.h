#ifndef _BCD_UTILS_H_
#define _BCD_UTILS_H_
#include "difacto/sarray.h"
namespace difacto {
namespace bcd {

/**
 * \brief partition the whole feature space into blocks
 *
 * @param fea_grp_nbits number of bit for encoding the feature group
 * @param fea_grps a list of (feature_group, num_partitions_this_group)
 * @param fea_blks a list of feature blocks with the start and end ID
 */
void PartitionFeatureSpace(int fea_grp_nbits,
                           const std::vector<std::pair<int,int>>& fea_grps,
                           std::vector<Range>* fea_blks) {

}

/**
 * \brief find the positionn of each feature block in the list of feature IDs
 *
 * @param fea_ids
 * @param fea_blks
 * @param positions
 */
void FindPosition(const SArray<feaid_t>& fea_ids,
                  const std::vector<Range>& fea_blks,
                  std::vector<Range>* positions) {

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


}  // namespace bcd
}  // namespace difacto
#endif  // _BCD_UTILS_H_
