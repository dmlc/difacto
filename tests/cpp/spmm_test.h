/**
 *  Copyright (c) 2015 by Contributors
 */
#include <cstring>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "difacto/sarray.h"
#include "common/range.h"
namespace difacto {
namespace test {

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
                    Vec* y,
                    int nt = DEFAULT_NTHREADS,
                    const Pos& x_pos = Pos(),
                    const Pos& y_pos = Pos()) {
    CHECK_NOTNULL(y);
    int dim = static_cast<int>(y->size() / D.size);
    Times(D, x.data(), y->data(), dim, nt);
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
                         Vec* y,
                         int nt = DEFAULT_NTHREADS,
                         const Pos& x_pos = Pos(),
                         const Pos& y_pos = Pos()) {
    TransTimes(D, x, 0, Vec(0), y, nt);
  }

  /**
   * \brief y = D^T * x + p * z
   * @param D n * m sparse matrix
   * @param x n * k length vector
   * @param p scalar
   * @param z m * k length vector,
   * @param y m * k length vector, should be pre-allocated
   * @param nthreads optional number of threads
   */
  template<typename Vec>
  static void TransTimes(const SpMat& D,
                         const Vec& x,
                         real_t p,
                         const Vec& z,
                         Vec* y,
                         int nt = DEFAULT_NTHREADS) {
    if (x.empty()) return;
    int dim = x.size() / D.size;
    if (z.size() == y->size() && p != 0) {
      TransTimes(D, x.data(), z.data(), p, y->data(), y->size(), dim, nt);
    } else {
      auto zero = x.data(); zero = NULL;
      TransTimes(D, x.data(), zero, static_cast<real_t>(0.0), y->data(), y->size(), dim, nt);
    }
  }

 private:
  // y = D * x
  template<typename V>
  static void Times(const SpMat& D, const V* const x,
                    V* y, int dim, int nt = DEFAULT_NTHREADS) {
    memset(y, 0, D.size * dim * sizeof(V));
#pragma omp parallel num_threads(nt)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V* y_i = y + i * dim;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            V const* x_j = x + D.index[j] * dim;
            V v = D.value[j];
            for (int k = 0; k < dim; ++k) y_i[k] += x_j[k] * v;
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            V const* x_j = x + D.index[j] * dim;
            for (int k = 0; k < dim; ++k) y_i[k] += x_j[k];
          }
        }
      }
    }
  }

  // y = D' * x
  template<typename V>
  static void TransTimes(const SpMat& D, const V* const x,
                         const V* const z, real_t p,
                         V* y, size_t y_size, int dim,
                         int nt = DEFAULT_NTHREADS) {
    if (z) {
      for (size_t i = 0; i < y_size; ++i) y[i] = z[i] * p;
    } else {
      memset(y, 0, y_size*sizeof(V));
    }

#pragma omp parallel num_threads(nt)
    {
      Range rg = Range(0, y_size/dim).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V const * x_i = x + i * dim;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned e = D.index[j];
            if (rg.Has(e)) {
              V v = D.value[j];
              V* y_j = y + e * dim;
              for (int k = 0; k < dim; ++k) y_j[k] += x_i[k] * v;
            }
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned e = D.index[j];
            if (rg.Has(e)) {
              V* y_j = y + e * dim;
              for (int k = 0; k < dim; ++k) y_j[k] += x_i[k];
            }
          }
        }
      }
    }
  }
};

}  // namespace test
}  // namespace difacto
