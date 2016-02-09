/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "./utils.h"
#include "difacto/base.h"
#include "common/spmt.h"
#include "data/localizer.h"

using namespace difacto;
TEST(SpMT, Transpose) {
  dmlc::data::RowBlockContainer<unsigned> data;
  std::vector<feaid_t> uidx;
  load_data(&data, &uidx);

  auto X = data.GetBlock();
  dmlc::data::RowBlockContainer<unsigned> Y, X2;
  SpMT::Transpose(X, &Y, uidx.size());
  SpMT::Transpose(Y.GetBlock(), &X2);

  auto X3 = X2.GetBlock();
  size_t nnz = X3.offset[X3.size];
  EXPECT_EQ(X3.size, X.size);
  EXPECT_EQ(norm1(X.offset, X.size+1),
            norm1(X3.offset, X.size+1));
  EXPECT_EQ(norm1(X.index, nnz),
            norm1(X3.index, nnz));
  EXPECT_EQ(norm2(X.value, nnz),
            norm2(X3.value, nnz));
}
