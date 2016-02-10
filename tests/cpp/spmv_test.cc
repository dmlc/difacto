/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "./utils.h"
#include "common/spmv.h"
#include "dmlc/timer.h"
#include "./spmv_test.h"

using namespace difacto;

namespace {
dmlc::data::RowBlockContainer<unsigned> data;
std::vector<feaid_t> uidx;
}  // namespace


TEST(SpMV, Times) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  SArray<real_t> x;
  gen_vals(uidx.size(), -10, 10, &x);

  SArray<real_t> y1(D.size);
  SArray<real_t> y2(D.size);

  test::SpMV::Times(D, x, &y1);
  SpMV::Times(D, x, &y2);
  EXPECT_EQ(norm2(y1), norm2(y2));
}

TEST(SpMV, TransTimes) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  SArray<real_t> x;
  gen_vals(D.size, -10, 10, &x);

  SArray<real_t> y1(uidx.size());
  SArray<real_t> y2(uidx.size());

  test::SpMV::TransTimes(D, x, &y1);
  SpMV::TransTimes(D, x, &y2);
  EXPECT_EQ(norm2(y1), norm2(y2));
}

TEST(SpMV, TimesPos) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  SArray<real_t> x;
  gen_vals(uidx.size(), -10, 10, &x);
  SArray<int> x_pos;
  SArray<real_t> x_val;
  test::gen_sliced_vec(x, &x_val, &x_pos);

  SArray<real_t> y(D.size);
  SArray<int> y_pos;
  SArray<real_t> y_val;
  test::gen_sliced_vec(y, &y_val, &y_pos);
  memset(y_val.data(), 0, y_val.size()*sizeof(real_t));

  test::SpMV::Times(D, x, &y);
  SpMV::Times(D, x_val, &y_val, DEFAULT_NTHREADS, x_pos, y_pos);

  EXPECT_EQ(norm2(y), norm2(y_val));

  SArray<real_t> y2;
  test::slice_vec(y_val, y_pos, &y2);
  EXPECT_EQ(norm2(y), norm2(y2));
}

TEST(SpMV, TransTimesPos) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  SArray<real_t> x;
  gen_vals(D.size, -10, 10, &x);
  SArray<int> x_pos;
  SArray<real_t> x_val;
  test::gen_sliced_vec(x, &x_val, &x_pos);

  SArray<real_t> y(uidx.size());
  SArray<int> y_pos;
  SArray<real_t> y_val;
  test::gen_sliced_vec(y, &y_val, &y_pos);
  memset(y_val.data(), 0, y_val.size()*sizeof(real_t));

  test::SpMV::TransTimes(D, x, &y);
  SpMV::TransTimes(D, x_val, &y_val, DEFAULT_NTHREADS, x_pos, y_pos);
  EXPECT_EQ(norm2(y), norm2(y_val));

  SArray<real_t> y2;
  test::slice_vec(y_val, y_pos, &y2);
  EXPECT_EQ(norm2(y), norm2(y2));
}
