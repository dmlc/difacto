#ifndef _TILE_BUILDER_H_
#define _TILE_BUILDER_H_
#include "common/kv_union.h"
#include "./tile_store.h"
#include "common/spmt.h"
#include "common/localizer.h"
#include "./bcd_utils.h"
namespace difacto {
namespace bcd {

/**
 * \brief preprocess data
 */
class TileBuilder {
 public:
  TileBuilder(TileStore* store) {
    store_ = store;
  }

  /**
   * \brief add a data block
   */
  void Add(const dmlc::RowBlock<feaid_t>& rowblk) {
    // map feature id into continous intergers and transpose to easy slice a
    // column block
    std::shared_ptr<std::vector<feaid_t>> ids(new std::vector<feaid_t>());
    std::shared_ptr<std::vector<real_t>> cnt(new std::vector<real_t>());
    auto compacted = new dmlc::data::RowBlockContainer<unsigned>();
    auto transposed = new dmlc::data::RowBlockContainer<unsigned>();

    Localizer lc(-1, nthreads_);
    lc.Compact(rowblk, compacted, ids.get(), cnt.get());
    SpMT::Transpose(compacted->GetBlock(), transposed, ids->size(), nthreads_);
    delete compacted;

    // store into tile store
    int id = blk_feaids_.size();
    SharedRowBlockContainer<unsigned> data(&transposed);
    store_->data_->Store(std::to_string(id) + "_data", data);
    store_->data_->Store(std::to_string(id) + "_label", rowblk.label, rowblk.size);
    delete transposed;

    // merge ids and counts
    SArray<feaid_t> sids(ids);
    SArray<real_t> scnt(cnt);
    if (feaids.empty()) {
      feaids = sids;
      feacnts = scnt;
    } else {
      SArray<feaid_t> new_feaids;
      SArray<real_t> new_feacnts;
      KVUnion(sids, scnt, feaids, feacnts,
              &new_feaids, &new_feacnts, 1, PLUS, nthreads_);
      feaids = new_feaids;
      feacnts = new_feacnts;
    }
    blk_feaids_.push_back(sids);
  }


  /**
   * \brief add filtered feature ids
   */
  void BuildColmap(const std::vector<Range>& feablk_range) {
    SArray<int> map(feaids.size());
    for (size_t i = 0; i < map.size(); ++i) {
      map[i] = i+1;
    }

    for (int i = 0; i < blk_feaids_.size(); ++i) {
      // store position
      std::vector<Range> pos;
      bcd::FindPosition(blk_feaids_[i], feablk_range, &pos);
      store_->colblk_pos_.push_back(pos);

      // store colmap
      SArray<int> colmap;
      KVMatch(feaids, map, blk_feaids_[i], &colmap, 1, ASSIGN, nthreads_);
      for (int& c : colmap) --c;  // unmatched will get -1
      store_->data_->Store(std::to_string(i) + "_colmap", colmap);

      // clear
      blk_feaids_[i].clear();
    }
    feaids.clear();
  }

  SArray<feaid_t> feaids;
  SArray<real_t> feacnts;

 private:
  std::vector<SArray<feaid_t>> blk_feaids_;
  TileStore* store_;
  int nthreads_ = DEFAULT_NTHREADS;
};

}  // namespace bcd
}  // namespace difacto
#endif  // _TILE_BUILDER_H_
