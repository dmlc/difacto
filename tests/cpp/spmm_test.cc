#include <gtest/gtest.h>
#include "./utils.h"
#include "common/spmv.h"
#include "common/spmm.h"
#include "dmlc/timer.h"
#include "./spmv_test.h"
#include "./spmm_test.h"

using namespace difacto;

dmlc::data::RowBlockContainer<unsigned> data;
std::vector<feaid_t> uidx;

TEST(SpMM, Times) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  int k = 10;
  SArray<real_t> x;
  gen_vals(uidx.size()*k, -10, 10, &x);

  SArray<real_t> y1(D.size*k);
  SArray<real_t> y2(D.size*k);

  test::SpMM::Times(D, x, &y1);
  SpMM::Times(D, x, k, &y2);
  EXPECT_EQ(norm2(y1), norm2(y2));
}

TEST(SpMM, TransTimes) {
  load_data(&data, &uidx);
  auto D = data.GetBlock();
  SArray<real_t> x;
  int k = 10;
  gen_vals(D.size*k, -10, 10, &x);

  SArray<real_t> y1(uidx.size()*k);
  SArray<real_t> y2(uidx.size()*k);

  test::SpMM::TransTimes(D, x, &y1);
  SpMM::TransTimes(D, x, k, &y2);
  EXPECT_EQ(norm2(y1), norm2(y2));
}

TEST(SpMM, TimesPosVec) {
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
  SArray<real_t> y_val2(y_val.size());

  SpMM::Times(D, x_val, 1, &y_val2, DEFAULT_NTHREADS, x_pos, y_pos);
  SpMV::Times(D, x_val, &y_val, DEFAULT_NTHREADS, x_pos, y_pos);

  EXPECT_EQ(norm2(y_val), norm2(y_val2));
}


TEST(SpMM, TransTimesPosVec) {
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
  SArray<real_t> y_val2(y_val.size());

  SpMM::TransTimes(D, x_val, 1, &y_val2, DEFAULT_NTHREADS, x_pos, y_pos);
  SpMV::TransTimes(D, x_val, &y_val, DEFAULT_NTHREADS, x_pos, y_pos);
  EXPECT_EQ(norm2(y_val), norm2(y_val2));
}
