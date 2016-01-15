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
   * \brief y = D * x
   *
   * @param D n * m sparse matrix
   * @param x m-length vector
   * @param y n-length vector, should be pre-allocated
   * @param nthreads optional number of threads
   * @tparam Vec can be either std::vector<T> or SArray<T>
   */
  template<typename Vec>
  static void Times(const SpMat& D, const Vec& x,
                    Vec* y, int nthreads = DEFAULT_NTHREADS) {
    CHECK_NOTNULL(y);
    CHECK_EQ(y->size(), D.size);
    Times(D, x.data(), y->data(), nthreads);
  }

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
  template<typename Vec, typename Pos>
  static void Times(const SpMat& D,
                    const Vec& x,
                    Vec* y,
                    int nthreads = DEFAULT_NTHREADS,
                    const Pos& x_pos = Vec(),
                    const Pos& y_pos = Vec()) {
    CHECK_NOTNULL(y);
    CHECK_EQ(y->size(), D.size);
    Times(D, x.data(), y->data(), nthreads);
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
  template<typename Vec, typename Pos>
  static void TransTimes(const SpMat& D,
                         const Vec& x,
                         Vec* y,
                         int nthreads = DEFAULT_NTHREADS,
                         const Pos& x_pos = Vec(),
                         const Pos& y_pos = Vec()) {
  }



  /**
   * \brief y = D^T * x
   * @param D n * m sparse matrix
   * @param x length n vector
   * @param y length m vector, should be pre-allocated
   * @param nthreads optional number of threads
   * @tparam Vec can be either std::vector<T> or SArray<T>
   */
  template<typename Vec>
  static void TransTimes(const SpMat& D, const Vec& x,
                    Vec* y, int nthreads = DEFAULT_NTHREADS) {
    CHECK_EQ(x.size(), D.size);
    CHECK_NOTNULL(y);
    TransTimes(D, x.data(), y->data(), y->size(), nthreads);
  }

 private:
  /** \brief y = D * x */
  template<typename V>
  static void Times(const SpMat& D,  const V* const x, V* y,
                    int nthreads = DEFAULT_NTHREADS) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V y_i = 0;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            y_i += x[D.index[j]] * D.value[j];
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j)
            y_i += x[D.index[j]];
        }
        y[i] = y_i;
      }
    }
  }

  /** \brief y = D^T * x */
  template<typename V>
  static void TransTimes(const SpMat& D,  const V* const x, V* y, size_t y_size,
                         int nthreads = DEFAULT_NTHREADS) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, y_size).Segment(
          omp_get_thread_num(), omp_get_num_threads());
      std::memset(y + rg.begin, 0, sizeof(V) * (rg.end - rg.begin));

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V x_i = x[i];
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i * D.value[j];
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i;
          }
        }
      }
    }
  }
};

}  // namespace difacto
#endif  // DIFACTO_COMMON_SPMV_H_
