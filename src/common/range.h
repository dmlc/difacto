/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_RANGE_H_
#define DIFACTO_COMMON_RANGE_H_
#include "dmlc/logging.h"
namespace difacto {
/**
 * \brief a range between [begin, end)
 */
struct Range {
  Range(uint64_t _begin, uint64_t _end) : begin(_begin), end(_end) { }
  Range() : Range(0, 0) { }
  ~Range() { }
  /**
   * \brief evenly divide this range into npart segments, and return the idx-th
   * one
   */
  inline Range Segment(uint64_t idx, uint64_t nparts) const {
    CHECK_GE(end, begin);
    CHECK_GT(nparts, (uint64_t)0);
    CHECK_LT(idx, nparts);
    double itv = static_cast<double>(end - begin) /
                 static_cast<double>(nparts);
    uint64_t _begin = static_cast<uint64_t>(begin + itv * idx);
    uint64_t _end = (idx == nparts - 1) ?
                  end : static_cast<uint64_t>(begin + itv * (idx+1));
    return Range(_begin, _end);
  }

  /**
   * \brief Return true if i contains in this range
   */
  inline bool Has(uint64_t i) const {
    return (begin <= i && i < end);
  }

  /**
   * \brief return a range for the whole range
   */
  static Range All() { return Range(0, -1); }

  inline bool Valid() const { return end > begin; }

  inline uint64_t Size() const { return end - begin; }

  bool operator== (const Range& rhs) const {
    return (begin == rhs.begin && end == rhs.end);
  }
  bool operator!= (const Range& rhs) const {
    return !(*this == rhs);
  }

  Range operator+ (const uint64_t v) const { return Range(begin+v, end+v); }
  Range operator- (const uint64_t v) const { return Range(begin-v, end-v); }
  Range operator* (const uint64_t v) const { return Range(begin*v, end*v); }

  uint64_t begin;
  uint64_t end;
};

}  // namespace difacto
#endif  // DIFACTO_COMMON_RANGE_H_
