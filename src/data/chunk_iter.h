/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_DATA_CHUNK_ITER_H_
#define DIFACTO_DATA_CHUNK_ITER_H_
#include <string>
#include <vector>
#include "difacto/base.h"
#include "dmlc/data.h"
#include "data/parser.h"
#include "data/libsvm_parser.h"
#include "./adfea_parser.h"
#include "./crb_parser.h"
#include "./criteo_parser.h"
namespace difacto {
/**
 * \brief an iterator reads a chunk with a hint data size
 */
class ChunkIter {
 public:
  /**
   * \brief create a chunk iterator
   *
   * @param uri filename
   * @param format the data format, support libsvm, crb, ...
   * @param part_index the i-th part to read
   * @param num_parts partition the file into serveral parts
   * @param chunk_size the chunk size.
   */
  ChunkIter(const std::string& uri,
            const std::string& format,
            unsigned part_index,
            unsigned num_parts,
            unsigned chunk_size) {
    char const* c_uri = uri.c_str();
    dmlc::InputSplit* input = dmlc::InputSplit::Create(
        c_uri, part_index, num_parts, format == "cb" ? "recordio" : "text");
    input->HintChunkSize(chunk_size);

    if (format == "libsvm") {
      parser_ = new dmlc::data::LibSVMParser<feaid_t>(input, 1);
    } else if (format == "criteo") {
      parser_ = new CriteoParser(input, true);
    } else if (format == "criteo_test") {
      parser_ = new CriteoParser(input, false);
    } else if (format ==  "adfea") {
      parser_ = new AdfeaParser(input);
    } else if (format == "rec") {
      parser_ = new CRBParser(input);
    } else {
      LOG(FATAL) << "unknown format " << format;
    }
    parser_ = new dmlc::data::ThreadedParser<feaid_t>(parser_);
  }

  ~ChunkIter() {
    delete parser_;
  }

  bool Next() {
    return parser_->Next();
  }

  const dmlc::RowBlock<feaid_t>& Value() const {
    return parser_->Value();
  }
 private:
  dmlc::data::ParserImpl<feaid_t> *parser_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_CHUNK_ITER_H_
