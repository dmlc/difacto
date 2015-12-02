#pragma once
#include "difacto/base.h"
namespace difacto {

/**
 * @brief Compact a rowblock's feature indices
 */
class Localizer {
 public:
  Localizer(int nthreads = 2) : nt_(nthreads) { }
  ~Localizer() { }

  /**
   * \brief compact blk's feature indices
   *
 Mapping a RowBlock from arbitrary feature index into continuous
 * feature indices starting from 0
   *
   * @param blk
   * @param compacted
   * @param uniq_idx
   * @param idx_frq
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
   * temporal results will be stored to accelerate \ref RemapIndex.
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
