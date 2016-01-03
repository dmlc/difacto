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
}  // namespace bcd
}  // namespace difacto
#endif  // _BCD_UTILS_H_
