/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_SPMM_H_
#define DIFACTO_COMMON_SPMM_H_
#include <cstring>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "difacto/sarray.h"
#include "./range.h"
namespace difacto {
/**
 * \brief multi-thread sparse matrix dense matrix multiplication
 *
 * comparing to \ref SpMV, the different is that both x and y are n-by-k matrices
 * rather than length-n vectors
 */
class SpMM {
 public:
  /** \brief row major sparse matrix */
  using SpMat = dmlc::RowBlock<unsigned>;
  /**
   * \brief y = D * x
   * @param D n * m sparse matrix
   * @param x m * k matrix
   * @param y n * k matrix, should be pre-allocated
   * @param nthreads optional number of threads
   * @param x_pos optional, the position of x's rows
   * @param y_pos optional, the position of y's rows
   * @tparam Vec can be either std::vector<T> or SArray<T>
   * @tparam Pos can be either std::vector<int> or SArray<int>
   */
  template<typename Vec, typename Pos = std::vector<int>>
  static void Times(const SpMat& D,
                    const Vec& x,
                    int k,
                    Vec* y,
                    int nthreads = DEFAULT_NTHREADS,
                    const Pos& x_pos = Pos(),
                    const Pos& y_pos = Pos()) {
    CHECK_NOTNULL(y);
    if (y_pos.size()) {
      CHECK_EQ(y_pos.size(), D.size);
    } else {
      CHECK_EQ(y->size(), D.size * k);
    }
    CheckPos(x_pos, x.size(), k);
    CheckPos(y_pos, y->size(), k);

    Times(D, x.data(), y->data(),
          (x_pos.empty() ? nullptr : x_pos.data()),
          (y_pos.empty() ? nullptr : y_pos.data()),
          k, nthreads);
  }

  /**
   * \brief y = D^T * x
   * @param D n * m sparse matrix
   * @param x n * k length vector
   * @param y m * k length vector, should be pre-allocated
   * @param nthreads optional number of threads
   * @tparam Vec can be either std::vector<T> or SArray<T>
   */
  template<typename Vec, typename Pos = std::vector<int>>
  static void TransTimes(const SpMat& D,
                         const Vec& x,
                         int k,
                         Vec* y,
                         int nthreads = DEFAULT_NTHREADS,
                         const Pos& x_pos = Pos(),
                         const Pos& y_pos = Pos()) {
    if (x_pos.size()) {
      CHECK_EQ(x_pos.size(), D.size);
    } else {
      CHECK_EQ(x.size(), D.size * k);
    }
    CHECK_NOTNULL(y);
    CheckPos(x_pos, x.size(), k);
    CheckPos(y_pos, y->size(), k);
    CHECK_GT(k, 0);
    size_t ncols = y_pos.size() ? y_pos.size() : y->size() / k;
    TransTimes(D, x.data(), y->data(),
               (x_pos.empty() ? nullptr : x_pos.data()),
               (y_pos.empty() ? nullptr : y_pos.data()),
               k, ncols, nthreads);
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
                    int k,
                    int nthreads) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V* y_i = GetPtr(y, y_pos, i, k);
        if (!y_i) continue;
        for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
          V const* x_j = GetPtr(x, x_pos, D.index[j], k);
          if (!x_j) continue;
          if (D.value) {
            V v = D.value[j];
            for (int l = 0; l < k; ++l) y_i[l] += x_j[l] * v;
          } else {
            for (int l = 0; l < k; ++l) y_i[l] += x_j[l];
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
                         I const* x_pos,
                         I const* y_pos,
                         int k,
                         size_t ncols,
                         int nthreads) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, ncols).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V const* x_i = GetPtr(x, x_pos, i, k);
        if (!x_i) continue;
        for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
          unsigned e = D.index[j];
          if (!rg.Has(e)) continue;
          V* y_j = GetPtr(y, y_pos, e, k);
          if (!y_j) continue;
          if (D.value) {
            V v = D.value[j];
            for (int l = 0; l < k; ++l) y_j[l] += x_i[l] * v;
          } else {
            for (int l = 0; l < k; ++l) y_j[l] += x_i[l];
          }
        }
      }
    }
  }

  template <typename V, typename I>
  static inline V* GetPtr(V* val, I const* pos, size_t idx, int k) {
    if (pos) {
      I pos_i = pos[idx];
      return pos_i == static_cast<I>(-1) ? nullptr : val+pos_i;
    } else {
      return val+idx*k;
    }
  }

  template<typename Pos>
  static inline void CheckPos(const Pos& pos, size_t max_len, int k) {
    for (auto p : pos) {
      size_t sp = static_cast<size_t>(p);
      if (sp != static_cast<size_t>(-1)) {
        CHECK_GE(sp, static_cast<size_t>(0));
        CHECK_LE(sp + k, max_len);
      }
    }
  }

};

}  // namespace difacto
#endif  // DIFACTO_COMMON_SPMM_H_
