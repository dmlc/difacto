/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_SPMV_H_
#define DIFACTO_COMMON_SPMV_H_
#include <cstring>
#include <vector>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "./range.h"
namespace difacto {

/**
 * \brief multi-thread sparse matrix vector multiplication
 */
class SpMV {
 public:
  /** \brief row major sparse matrix */
  using SpMat = dmlc::RowBlock<unsigned>;
  /**
   * \brief y += D * x
   *
   * both x and y are vectors. beside the normal vector format, one can specify
   * an optional entry position to slice a vector from another vector. the
   * following two representation are equal
   *
   * \code
   * a = {1, 3, 0, 5};
   * \endcode
   *
   * \code
   * a = {1, 2, 3, 4, 5, 6};
   * a_pos = {0, 2, -1, 4};
   * \endcode
   *
   * here position -1 means empty
   * - if a is x, then means value 0
   * - if a is y, the result will be not written into y
   *
   * @param D n * m sparse matrix
   * @param x vector x
   * @param y vector y, should be pre-allocated
   * @param nthreads optional, number of threads
   * @param x_pos optional, the position of x
   * @param y_pos optional, the position of y
   * @tparam Vec can be either std::vector<T> or SArray<T>
   * @tparam Pos can be either std::vector<int> or SArray<int>
   */
  template<typename Vec, typename Pos = std::vector<int>>
  static void Times(const SpMat& D,
                    const Vec& x,
                    Vec* y,
                    int nthreads = DEFAULT_NTHREADS,
                    const Pos& x_pos = Pos(),
                    const Pos& y_pos = Pos()) {
    CHECK_NOTNULL(y);
    if (y_pos.size()) {
      CHECK_EQ(y_pos.size(), D.size);
    } else {
      CHECK_EQ(y->size(), D.size);
    }
    CheckPos(x_pos, x.size());
    CheckPos(y_pos, y->size());
    Times(D, x.data(), y->data(),
          (x_pos.empty() ? nullptr : x_pos.data()),
          (y_pos.empty() ? nullptr : y_pos.data()),
          nthreads);
  }

  /**
   * \brief y += D^T * x
   *
   * @param D n * m sparse matrix
   * @param x vector x
   * @param y vector y, should be pre-allocated
   * @param nthreads optional, number of threads
   * @param x_pos optional, the position of x
   * @param y_pos optional, the position of y
   * @tparam Vec can be either std::vector<T> or SArray<T>
   * @tparam Pos can be either std::vector<int> or SArray<int>
   */
  template<typename Vec, typename Pos = std::vector<int>>
  static void TransTimes(const SpMat& D,
                         const Vec& x,
                         Vec* y,
                         int nthreads = DEFAULT_NTHREADS,
                         const Pos& x_pos = Pos(),
                         const Pos& y_pos = Pos()) {
    if (x_pos.size()) {
      CHECK_EQ(x_pos.size(), D.size);
    } else {
      CHECK_EQ(x.size(), D.size);
    }
    CHECK_NOTNULL(y);
    CheckPos(x_pos, x.size());
    CheckPos(y_pos, y->size());
    TransTimes(D, x.data(), y->data(), y->size(),
               (x_pos.empty() ? nullptr : x_pos.data()),
               (y_pos.empty() ? nullptr : y_pos.data()),
               nthreads);
  }

 private:
  /**
   * \brief y += D * x, C pointer version
   */
  template<typename V, typename I>
  static void Times(const SpMat& D,
                    V const* x,
                    V* y,
                    I const* x_pos,
                    I const* y_pos,
                    int nthreads) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V* y_i = GetPtr(y, y_pos, i);
        if (!y_i) continue;
        for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
          V x_j = GetVal(x, x_pos, D.index[j]);
          if (x_j == 0) continue;
          if (D.value) {
            *y_i += x_j * D.value[j];
          } else {
            *y_i += x_j;
          }
        }
      }
    }
  }

  /**
   * \brief y += D' * x, C pointer version
   */
  template<typename V, typename I>
  static void TransTimes(const SpMat& D,
                         V const* x,
                         V* y,
                         size_t y_size,
                         I const* x_pos,
                         I const* y_pos,
                         int nthreads) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, y_size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V x_i = GetVal(x, x_pos, i);
        if (x_i == 0) continue;
        for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
          unsigned k = D.index[j];
          if (rg.Has(k)) {
            V* y_j = GetPtr(y, y_pos, k);
            if (y_j) {
              if (D.value) {
                *y_j += x_i * D.value[j];
              } else {
                *y_j += x_i;
              }
            }
          }
        }
      }
    }
  }

  template <typename V, typename I>
  static inline V GetVal(V const* val, I const* pos, size_t idx) {
    if (pos) {
      I pos_i = pos[idx];
      return pos_i == static_cast<I>(-1) ? 0 : val[pos_i];
    } else {
      return val[idx];
    }
  }

  template <typename V, typename I>
  static inline V* GetPtr(V* val, I const* pos, size_t idx) {
    if (pos) {
      I pos_i = pos[idx];
      return pos_i == static_cast<I>(-1) ? nullptr : val+pos_i;
    } else {
      return val+idx;
    }
  }

  template<typename Pos>
  static inline void CheckPos(const Pos& pos, size_t max_len) {
    for (auto p : pos) {
      size_t sp = static_cast<size_t>(p);
      if (sp != static_cast<size_t>(-1)) {
        CHECK_GE(sp, static_cast<size_t>(0));
        CHECK_LT(sp, max_len);
      }
    }
  }
};

}  // namespace difacto
#endif  // DIFACTO_COMMON_SPMV_H_
