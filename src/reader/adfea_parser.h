/**
 * Copyright (c) 2015 by Contributors
 * @file   adfea_parser.h
 * @brief  parse adfea ctr data format
 */
#ifndef DIFACTO_READER_ADFEA_PARSER_H_
#define DIFACTO_READER_ADFEA_PARSER_H_
#include <limits>
#include <vector>
#include "difacto/base.h"
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
namespace difacto {

/**
 * \brief adfea ctr dataset
 * the top 10 bits store the feature group id
 */
class AdfeaParser : public dmlc::data::ParserImpl<feaid_t> {
 public:
  explicit AdfeaParser(dmlc::InputSplit *source)
      : bytes_read_(0), source_(source) { }
  virtual ~AdfeaParser() {
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
    using dmlc::data::isspace;
    using dmlc::data::isdigit;
    using dmlc::data::strtoull;

    dmlc::InputSplit::Blob chunk;

    if (!source_->NextChunk(&chunk)) return false;

    CHECK_NE(chunk.size, 0);
    bytes_read_ += chunk.size;
    data->resize(1);
    dmlc::data::RowBlockContainer<feaid_t>& blk = (*data)[0];
    blk.Clear();
    int i = 0;
    char *p = reinterpret_cast<char*>(chunk.dptr);
    char *end = p + chunk.size;

    while (isspace(*p) && p != end) ++p;
    while (p != end) {
      char *head = p;
      while (isdigit(*p) && p != end) ++p;
      CHECK_NE(head, p);

      if (*p == ':') {
        ++p;
        feaid_t idx = strtoull(head, NULL, 10);
        feaid_t gid = strtoull(p, NULL, 10);
        blk.index.push_back(EncodeFeaGrpID(idx, gid, 12));
        while (isdigit(*p) && p != end) ++p;
      } else {
        // skip the lineid and the first count
        if (i == 2) {
          i = 0;
          if (blk.label.size() != 0) {
            blk.offset.push_back(blk.index.size());
          }
          blk.label.push_back(*head == '1');
        } else {
          ++i;
        }
      }

      while (isspace(*p) && p != end) ++p;
    }
    if (blk.label.size() != 0) {
      blk.offset.push_back(blk.index.size());
    }
    return true;
  }

 private:
  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  dmlc::InputSplit *source_;
};

}  // namespace difacto
#endif  // DIFACTO_READER_ADFEA_PARSER_H_
