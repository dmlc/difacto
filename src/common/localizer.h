#ifndef DIFACTO_LOCALIZER_H_
#define DIFACTO_LOCALIZER_H_
#include <vector>
#include "difacto/base.h"
#include "dmlc/io.h"
#include "data/row_block.h"
namespace difacto {

/**
 * @brief Compact a rowblock's feature indices
 */
class Localizer {
 public:
  /**
   * \brief constructor
   *
   * \param max_index feature index will be projected into [0, max_index) by mod
   * \param nthreads number of threads
   */

  Localizer(feaid_t max_index = std::numeric_limits<feaid_t>::max(),
            int nthreads = 2)
      : max_index_(max_index), nt_(nthreads) { }
  ~Localizer() { }

  /**
   * \brief compact blk's feature indices
   *
   * This function maps a RowBlock from arbitrary feature index into continuous
   * feature indices starting from 0
   *
   * @param blk the data block
   * @param compacted the new block with feature index remapped
   * @param uniq_idx if not null, then return the original unique feature indices
   * @param idx_frq if not null, then return the according feature occurance
   */
  void Compact(const dmlc::RowBlock<feaid_t>& blk,
               dmlc::data::RowBlockContainer<unsigned> *compacted,
               std::vector<feaid_t>* uniq_idx = NULL,
               std::vector<real_t>* idx_frq = NULL) {
    std::vector<feaid_t>* uidx =
        uniq_idx == NULL ? new std::vector<feaid_t>() : uniq_idx;
    CountUniqIndex(blk, uidx, idx_frq);
    RemapIndex(blk, *uidx, compacted);
    if (uniq_idx == NULL) delete uidx;
    Clear();
  }

  /**
   * @brief find the unique indices and count the occurance
   *
   * This function stores temporal results to accelerate \ref RemapIndex.
   *
   * @param idx the item list in any order
   * @param uniq_idx returns the sorted unique items
   * @param idx_frq if not NULL then returns the according occurrence counts
   */
  void CountUniqIndex(const dmlc::RowBlock<feaid_t>& blk,
                      std::vector<feaid_t>* uniq_idx,
                      std::vector<real_t>* idx_frq);

  /**
   * @brief Remaps the index.
   *
   * @param idx_dict the index dictionary, which should be ordered. Any index
   * does not exists in this dictionary is dropped.
   *
   * @param compacted a rowblock with index mapped: idx_dict[i] -> i.
   */
  void RemapIndex(const dmlc::RowBlock<feaid_t>& blk,
                  const std::vector<feaid_t>& idx_dict,
                  dmlc::data::RowBlockContainer<unsigned> *compacted);

  /**
   * @brief Clears the temporal results
   */
  void Clear() { pair_.clear(); }

 private:
  feaid_t max_index_;
  /** \brief number of threads */
  int nt_;

#pragma pack(push)
#pragma pack(4)
  struct Pair {
    feaid_t k; unsigned i;
  };
#pragma pack(pop)
  std::vector<Pair> pair_;
};
}  // namespace difacto

#endif  // DIFACTO_LOCALIZER_H_
