/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_COMMON_SPMT_H_
#define DIFACTO_COMMON_SPMT_H_
#include <cstring>
#include <vector>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "./range.h"
#include "data/row_block.h"
namespace difacto {

/**
 * \brief multi-thread sparse matrix transpose
 */
class SpMT {
 public:
  /**
   * \brief transpose matrix Y = X'
   * \param X sparse matrix in row major
   * \param Y sparse matrix in row major
   * \param X_ncols optional, number of columns in X
   * \param nt optional, number of threads
   */
  static void Transpose(const dmlc::RowBlock<unsigned>& X,
                        dmlc::data::RowBlockContainer<unsigned>* Y,
                        unsigned X_ncols = 0,
                        int nt = DEFAULT_NTHREADS) {
    // find number of columns in X
    size_t nrows = X.size;
    size_t nnz = X.offset[nrows] - X.offset[0];
    if (X_ncols == 0) {
      for (size_t i = 0; i < nnz; ++i) {
        if (X_ncols < X.index[i]) X_ncols = X.index[i];
      }
      ++X_ncols;
    }

    // allocate Y
    CHECK_NOTNULL(Y);
    Y->offset.clear();
    Y->offset.resize(X_ncols+1, 0);
    Y->index.resize(nnz);
    if (X.value) Y->value.resize(nnz);

    // fill Y->offset
#pragma omp parallel num_threads(nt)
    {
      Range range = Range(0, X_ncols).Segment(
          omp_get_thread_num(), omp_get_num_threads());
      for (size_t i = 0; i < nnz; ++i) {
        unsigned k = X.index[i];
        if (!range.Has(k)) continue;
        ++Y->offset[k+1];
      }
    }
    for (size_t i = 0; i < X_ncols; ++i) {
      Y->offset[i+1] += Y->offset[i];
    }

    // fill Y->index and Y->value
#pragma omp parallel num_threads(nt)
    {
      Range range = Range(0, X_ncols).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = 0; i < nrows; ++i) {
        if (X.offset[i] == X.offset[i+1]) continue;
        for (size_t j = X.offset[i]; j < X.offset[i+1]; ++j) {
          unsigned k = X.index[j];
          if (!range.Has(k)) continue;
          if (X.value) {
            Y->value[Y->offset[k]] = X.value[j];
          }
          Y->index[Y->offset[k]] = static_cast<unsigned>(i);
          ++Y->offset[k];
        }
      }
    }

    // restore Y->offset
    if (X_ncols > 0) {
      for (size_t i = X_ncols -1; i > 0; --i) {
        Y->offset[i] = Y->offset[i-1];
      }
      Y->offset[0] = 0;
    }
  }
};
}  // namespace difacto
#endif  // DIFACTO_COMMON_SPMT_H_
