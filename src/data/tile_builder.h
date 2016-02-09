/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_DATA_TILE_BUILDER_H_
#define DIFACTO_DATA_TILE_BUILDER_H_
#include "common/kv_union.h"
#include "common/spmt.h"
#include "data/localizer.h"
#include "./tile_store.h"
namespace difacto {
/**
 * \brief helper class to build TileStore
 */
class TileBuilder {
 public:
  TileBuilder(TileStore* store, int nthreads, bool allow_multi_columns = false) {
    store_ = store;
    nthreads_ = nthreads;
    multicol_ = allow_multi_columns;
  }

  /**
   * \brief add a raw rowblk to the store
   * feaids = feaids \cup new_feaids
   * feacnts = feacnts \cup new_feacnts
   */
  void Add(const dmlc::RowBlock<feaid_t>& rowblk,
           SArray<feaid_t>* feaids = nullptr,
           SArray<real_t>* feacnts = nullptr) {
    // map feature id into continous intergers
    std::shared_ptr<std::vector<feaid_t>> ids(new std::vector<feaid_t>());
    std::shared_ptr<std::vector<real_t>> cnts(new std::vector<real_t>());
    auto compacted = new dmlc::data::RowBlockContainer<unsigned>();
    Localizer lc(-1, nthreads_);
    lc.Compact(rowblk, compacted, ids.get(), feacnts ? cnts.get() : nullptr);

    // store data into tile store
    int id = blk_feaids_.size();
    if (multicol_) {
      // transpose to easy slice a column block
      auto transposed = new dmlc::data::RowBlockContainer<unsigned>();
      SpMT::Transpose(compacted->GetBlock(), transposed, ids->size(), nthreads_);
      delete compacted;
      SharedRowBlockContainer<unsigned> data(&transposed);
      data.label.CopyFrom(rowblk.label, rowblk.size);
      store_->Store(id, data);
      delete transposed;
    } else {
      SharedRowBlockContainer<unsigned> data(&compacted);
      store_->Store(id, data);
      delete compacted;
    }

    // store ids, which are used to build colmap
    SArray<feaid_t> sids(ids);
    blk_feaids_.push_back(sids);

    if (!feaids) return;
    CHECK_NOTNULL(feacnts);
    SArray<real_t> scnts(cnts);
    CHECK_EQ(feaids->size(), feacnts->size());
    CHECK_EQ(sids.size(), scnts.size());
    KVUnion(sids, scnts, feaids, feacnts);
  }

  /**
   * \brief build colmap
   * \param feaids
   */
  void BuildColmap(const SArray<feaid_t>& feaids,
                   const std::vector<Range>& feablk_range = {},
                   std::vector<Range> *feapos = nullptr) {
    SArray<int> map(feaids.size());
    for (size_t i = 0; i < map.size(); ++i) {
      map[i] = i+1;  // start from 1
    }
    store_->meta_.resize(blk_feaids_.size());
    for (size_t i = 0; i < blk_feaids_.size(); ++i) {
      // store colmap
      SArray<int> colmap;
      KVMatch(feaids, map, blk_feaids_[i], &colmap, ASSIGN, nthreads_);
      for (int& c : colmap) --c;  // unmatched will get -1
      store_->Store(i, colmap);

      // update meta
      std::vector<Range> pos;
      if (feablk_range.size()) {
        CHECK(multicol_) << "you should set allow_multi_columns = true";
        FindPosition(blk_feaids_[i], feablk_range, &pos);
      }

      auto key = std::to_string(i) + "_";
      if (pos.empty()) {
        TileStore::Meta c;
        c.colmap = Range(0, colmap.size());
        c.offset = Range(0, store_->data_->size(key+"offset"));
        c.index = Range(0, store_->data_->size(key+"index"));
        store_->meta_[i].push_back(c);
      } else {
        SArray<size_t> offset;
        store_->data_->Fetch(key+"offset", &offset);
        CHECK_EQ(offset.size(), colmap.size()+1);
        for (auto p : pos) {
          TileStore::Meta c;
          c.colmap = p;
          c.offset = Range(p.begin, p.end+1);
          c.index = Range(offset[p.begin], offset[p.end]);
          store_->meta_[i].push_back(c);
        }
      }
      // clear
      blk_feaids_[i].clear();
    }
    if (feapos) FindPosition(feaids, feablk_range, feapos);
  }

 private:
  /**
   * \brief find the positionn of each feature block in the list of feature IDs
   *
   * @param feaids
   * @param feablks
   * @param positions
   */
  static void FindPosition(const SArray<feaid_t>& feaids,
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
      auto ub = std::lower_bound(lb, end, feablks[i].end);
      cur = ub;
      positions->at(i) = Range(lb - begin, ub - begin);
    }
  }

  std::vector<SArray<feaid_t>> blk_feaids_;
  TileStore* store_;
  int nthreads_;
  bool multicol_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_TILE_BUILDER_H_
