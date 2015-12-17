/**
 * Copyright (c) 2015 by Contributors
 * @file   criteo_parser.h
 * @brief  parse criteo ctr data format
 */
#ifndef DIFACTO_DATA_CRITEO_PARSER_H_
#define DIFACTO_DATA_CRITEO_PARSER_H_
#include <limits>
#if DIFACTO_USE_CITY
#include <city.h>
#endif  // DIFACTO_USE_CITY
#include <vector>
#include "difacto/base.h"
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
namespace difacto {

/**
 * \brief criteo ctr dataset:
 * The columns are tab separeted with the following schema:
 *  <label> <integer feature 1> ... <integer feature 13>
 *  <categorical feature 1> ... <categorical feature 26>
 */
class CriteoParser : public dmlc::data::ParserImpl<feaid_t> {
 public:
  explicit CriteoParser(dmlc::InputSplit *source, bool is_train)
      : bytes_read_(0), source_(source), is_train_(is_train) {
  }
  virtual ~CriteoParser() {
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
    dmlc::InputSplit::Blob chunk;
    if (!source_->NextChunk(&chunk)) return false;

    CHECK_NE(chunk.size, 0);
    bytes_read_ += chunk.size;
    char *p = reinterpret_cast<char*>(chunk.dptr);
    char *end = p + chunk.size;
    data->resize(1);
    dmlc::data::RowBlockContainer<feaid_t>& blk = (*data)[0];
    blk.Clear();
    char *pp = p;
    while (p != end) {
      while (*p == '\r' || *p == '\n') ++p;
      if (p == end) break;

      // parse label
      if (is_train_) {
        pp = Find(p, end, '\t');
        CHECK_NE(p, pp) << "no label.., try criteo_test";
        blk.label.push_back(atof(p));
        p = pp + 1;
      } else {
        blk.label.push_back(0);
      }

      // parse inter feature
      for (feaid_t i = 0; i < 13; ++i) {
        pp = Find(p, end, '\t');
        CHECK_NOTNULL(pp);
        if (pp > p) {
          blk.index.push_back(
              (Hash(p, pp-p)>>10) | ((size_t)i << 54));
        }
        p = pp + 1;
      }

      // parse categorty feature
      for (int i = 0; i < 26; ++i) {
        if (p == end) break;
        if (isspace(*p)) { ++p; continue; }
        pp = p + 8; CHECK(isspace(*pp)) << i << " " << end - p << " " << *p;
        size_t len = pp - p;
        if (len) blk.index.push_back(
                (Hash(p, len)>>10) | ((size_t)(i+13) << 54));
        p = pp + 1;
        if (*pp == '\n' || *pp == '\r') break;
      }
      blk.offset.push_back(blk.index.size());
    }
    return true;
  }

 private:
  inline feaid_t Hash(const char* p, size_t len) {
#if DIFACTO_USE_CITY
    return CityHash64(p, len);
#else
    LOG(FATAL) << "compile with USE_CITY=1";
    return 0;
#endif  // DIFACTO_USE_CITY
  }

  // implement strchr
  inline char* Find(char* p, char* end, int c) {
    while (p != end && *p != c) ++p;
    return p;
  }

  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  dmlc::InputSplit *source_;
  bool is_train_;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_CRITEO_PARSER_H_
