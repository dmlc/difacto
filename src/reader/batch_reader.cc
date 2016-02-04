/**
 * Copyright (c) 2015 by Contributors
 */
#include "./batch_reader.h"
namespace difacto {

BatchReader::BatchReader(
    const std::string& uri, const std::string& format,
    unsigned part_index, unsigned num_parts,
    unsigned batch_size, unsigned shuffle_buf_size,
    float neg_sampling) {
  batch_size_   = batch_size;
  shuf_buf_    = shuffle_buf_size;
  neg_sampling_ = neg_sampling;
  start_        = 0;
  end_          = 0;
  seed_         = 0;
  if (shuf_buf_) {
    CHECK_GE(shuf_buf_, batch_size_);
    buf_reader_ = new BatchReader(
        uri, format, part_index, num_parts, shuf_buf_);
    reader_ = NULL;
  } else {
    buf_reader_ = NULL;
    reader_ = new Reader(uri, format, part_index, num_parts, 1<<26);
  }
}

bool BatchReader::Next() {
  batch_.Clear();
  while (batch_.offset.size() < batch_size_ + 1) {
    if (start_ == end_) {
      if (shuf_buf_ == 0) {
        // no random shuffle
        if (!reader_->Next()) break;
        in_blk_ = reader_->Value();
      } else {
        // do random shuffle
        if (!buf_reader_->Next()) break;
        in_blk_ = buf_reader_->Value();
        if (rdp_.size() != in_blk_.size) {
          rdp_.resize(in_blk_.size);
          for (size_t i = 0; i < in_blk_.size; ++i) rdp_[i] = i;
        }
        std::random_shuffle(rdp_.begin(), rdp_.end());
      }
      start_ = 0;
      end_ = in_blk_.size;
    }

    size_t len = std::min(end_ - start_, batch_size_ + 1 - batch_.offset.size());
    if (shuf_buf_ == 0 && neg_sampling_ == 1.0) {
      Push(start_, len);
    } else {
      for (size_t i = start_; i < start_ + len; ++i) {
        int j = rdp_[i];
        // downsampling
        float p = static_cast<float>(rand_r(&seed_)) /
                  static_cast<float>(RAND_MAX);
        if (neg_sampling_ < 1.0 &&
            in_blk_.label[j] <= 0 &&
            p > 1 - neg_sampling_) {
          continue;
        }
        batch_.Push(in_blk_[j]);
      }
    }
    start_ += len;
  }

  bool binary =  true;
  for (auto f : batch_.value) if (f != 1) { binary = false; break; }
  if (binary) batch_.value.clear();

  out_blk_ = batch_.GetBlock();

  return out_blk_.size > 0;
}

void BatchReader::Push(size_t pos, size_t len) {
  if (!len) return;
  CHECK_LE(pos + len, in_blk_.size);
  dmlc::RowBlock<feaid_t> slice;
  slice.weight = NULL;
  slice.size = len;
  slice.offset  = in_blk_.offset + pos;
  slice.label   = in_blk_.label  + pos;
  slice.index   = in_blk_.index  + in_blk_.offset[pos];
  if (in_blk_.value) {
    slice.value = in_blk_.value  + in_blk_.offset[pos];
  } else {
    slice.value = NULL;
  }
  batch_.Push(slice);
}

}  // namespace difacto
