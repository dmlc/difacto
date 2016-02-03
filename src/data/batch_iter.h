/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_DATA_BATCH_ITER_H_
#define DIFACTO_DATA_BATCH_ITER_H_
#include <string>
#include <vector>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "data/parser.h"
namespace difacto {

/**
 * \brief an iterator reads a batch with a given number of examples
 * each time.
 */
class BatchIter {
 public:
  /**
   * \brief create a batch iterator
   *
   * @param uri filename
   * @param format the data format, support libsvm, rec, ...
   * @param part_index the i-th part to read
   * @param num_parts partition the file into serveral parts
   * @param batch_size the batch size.
   * @param shuffle_size if nonzero, then the batch is randomly picked from a buffer with
   * shuffle_buf_size examples
   * @param neg_sampling the probability to pickup a negative sample (label <= 0)
   */
  BatchIter(const std::string& uri,
            const std::string& format,
            unsigned part_index,
            unsigned num_parts,
            unsigned batch_size,
            unsigned shuffle_buf_size = 0,
            float neg_sampling = 1.0);

  ~BatchIter() {
    delete parser_;
    delete buf_reader_;
  }

  /**
   * \brief read the next batch
   */
  bool Next();

  /**
   * \brief get the current batch
   *
   */
  const dmlc::RowBlock<feaid_t>& Value() const {
    return out_blk_;
  }

  /**
   * \brief reset to the file beginning
   */
  void Reset() {
    if (parser_) parser_->BeforeFirst();
    if (buf_reader_) buf_reader_->Reset();
  }

 private:
  /**
   * \brief batch_.push(in_blk_(pos:pos+len))
   */
  void Push(size_t pos, size_t len);

  unsigned batch_size_, shuf_buf_;
  dmlc::data::ParserImpl<feaid_t> *parser_;

  float neg_sampling_;
  size_t start_, end_;
  dmlc::RowBlock<feaid_t> in_blk_, out_blk_;
  dmlc::data::RowBlockContainer<feaid_t> batch_;

  // random pertubation
  std::vector<unsigned> rdp_;
  BatchIter* buf_reader_;
  unsigned int seed_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_BATCH_ITER_H_
