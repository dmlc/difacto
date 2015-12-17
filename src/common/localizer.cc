/**
 * Copyright (c) 2015 by Contributors
 */
#include "./localizer.h"
#include "dmlc/omp.h"
#include "dmlc/logging.h"
#include "./parallel_sort.h"
namespace difacto {

void Localizer::CountUniqIndex(
    const dmlc::RowBlock<feaid_t>& blk,
    std::vector<feaid_t>* uniq_idx,
    std::vector<real_t>* idx_frq) {
  // sort
  if (blk.size == 0) return;
  size_t idx_size = blk.offset[blk.size];
  CHECK_LT(idx_size, static_cast<size_t>(std::numeric_limits<unsigned>::max()))
      << "you need to change Pair.i from unsigned to uint64";
  pair_.resize(idx_size);

#pragma omp parallel for num_threads(nt_)
  for (size_t i = 0; i < idx_size; ++i) {
    pair_[i].k = ReverseBytes(blk.index[i] % max_index_);
    pair_[i].i = i;
  }

  ParallelSort(&pair_, nt_,
               [](const Pair& a, const Pair& b) {return a.k < b.k; });

  // save data
  CHECK_NOTNULL(uniq_idx);
  uniq_idx->clear();
  if (idx_frq) idx_frq->clear();

  feaid_t curr = pair_[0].k;
  real_t cnt = 0;
  for (size_t i = 0; i < pair_.size(); ++i) {
    const Pair& v = pair_[i];
    if (v.k != curr) {
      uniq_idx->push_back(curr);
      curr = v.k;
      if (idx_frq) idx_frq->push_back(cnt);
      cnt = 0;
    }
    ++cnt;
  }
  uniq_idx->push_back(curr);
  if (idx_frq) idx_frq->push_back(cnt);
}


void Localizer::RemapIndex(
    const dmlc::RowBlock<feaid_t>& blk,
    const std::vector<feaid_t>& idx_dict,
    dmlc::data::RowBlockContainer<unsigned> *compacted) {
  if (blk.size == 0 || idx_dict.empty()) return;
  CHECK_LT(idx_dict.size(),
           static_cast<size_t>(std::numeric_limits<unsigned>::max()));
  CHECK_EQ(blk.offset[blk.size], pair_.size());

  // build the index mapping
  unsigned matched = 0;
  std::vector<unsigned> remapped_idx(pair_.size(), 0);
  auto cur_dict = idx_dict.cbegin();
  auto cur_pair = pair_.cbegin();
  while (cur_dict != idx_dict.cend() && cur_pair != pair_.cend()) {
    if (*cur_dict < cur_pair->k) {
      ++cur_dict;
    } else {
      if (*cur_dict == cur_pair->k) {
        remapped_idx[cur_pair->i]
            = static_cast<unsigned>((cur_dict-idx_dict.cbegin()) + 1);
        ++matched;
      }
      ++cur_pair;
    }
  }

  // construct the new rowblock
  auto o = compacted;
  CHECK_NOTNULL(o);
  o->offset.resize(blk.size+1); o->offset[0] = 0;
  o->index.resize(matched);
  if (blk.value) o->value.resize(matched);

  size_t k = 0;
  for (size_t i = 0; i < blk.size; ++i) {
    for (size_t j = blk.offset[i]; j < blk.offset[i+1]; ++j) {
      if (remapped_idx[j] == 0) continue;
      if (blk.value) o->value[k] = blk.value[j];
      o->index[k++] = remapped_idx[j] - 1;
    }
    o->offset[i+1] = k;
  }
  CHECK_EQ(k, matched);

  if (blk.label) {
    o->label.resize(blk.size);
    memcpy(o->label.data(), blk.label, blk.size*sizeof(real_t));
  }
  o->max_index = idx_dict.size() - 1;
}

}  // namespace difacto
