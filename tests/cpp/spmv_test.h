/**
 * Copyright (c) 2015 by Contributors
 */
#include <cstring>
#include <vector>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "common/range.h"
namespace difacto {
namespace test {

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

}  // namespace test
}  // namespace difacto
