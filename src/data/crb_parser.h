/**
 * Copyright (c) 2015 by Contributors
 * @file   crb_parser.h
 * @brief  parser for compressed row block data format
 */
#ifndef DIFACTO_DATA_CRB_PARSER_H_
#define DIFACTO_DATA_CRB_PARSER_H_
#include <vector>
#include "data/parser.h"
#include "dmlc/recordio.h"
#include "./compressed_row_block.h"
namespace difacto {
/**
 * \brief compressed row block parser
 */
class CRBParser : public dmlc::data::ParserImpl<feaid_t> {
 public:
  explicit CRBParser(dmlc::InputSplit *source)
      : bytes_read_(0), source_(source) {
  }
  virtual ~CRBParser() {
    delete source_;
  }
  void BeforeFirst(void) override {
    source_->BeforeFirst();
  }
  size_t BytesRead(void) const override {
    return bytes_read_;
  }
  bool ParseNext(
      std::vector<dmlc::data::RowBlockContainer<feaid_t> > *data) override {
    dmlc::InputSplit::Blob rec;
    if (!source_->NextRecord(&rec)) return false;
    CHECK_NE(rec.size, 0);
    bytes_read_ += rec.size;
    data->resize(1); (*data)[0].Clear();
    CompressedRowBlock crb;
    crb.Decompress((char const*)rec.dptr, rec.size, &(*data)[0]);
    return true;
  }

 private:
  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  dmlc::InputSplit *source_;
};
}  // namespace difacto
#endif  // DIFACTO_DATA_CRB_PARSER_H_
