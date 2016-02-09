/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "common/spmt.h"
#include "loss/logit_loss_delta.h"
#include "loss/logit_loss.h"
#include "./utils.h"

using namespace difacto;

TEST(LogitLossDelta, Grad) {
  // load and tranpose data
  dmlc::data::RowBlockContainer<unsigned> rowblk, transposed;
  std::vector<feaid_t> uidx;
  load_data(&rowblk, &uidx);
  SpMT::Transpose(rowblk.GetBlock(), &transposed, uidx.size());

  // init loss
  KWArgs args = {{"compute_hession", "0"}};
  LogitLossDelta loss; loss.Init(args);
  LogitLoss ref_loss;

  for (int i = 0; i < 10; ++i) {
    SArray<real_t> w;
    gen_vals(uidx.size(), -10, 10, &w);

    SArray<real_t> ref_pred(100), ref_grad(w.size());
    ref_loss.Predict(rowblk.GetBlock(), {SArray<char>(w)}, &ref_pred);
    ref_loss.CalcGrad(rowblk.GetBlock(), {SArray<char>(ref_pred)}, &ref_grad);

    int nblk = 10;
    SArray<real_t> pred(100), grad(w.size());
    for (int b = 0; b < nblk; ++b) {
      auto rg = Range(0, w.size()).Segment(b, nblk);
      auto data = transposed.GetBlock().Slice(rg.begin, rg.end);
      data.label = rowblk.GetBlock().label;
      loss.Predict(data, {SArray<char>(w.segment(rg.begin, rg.end))}, &pred);
    }

    for (int b = 0; b < nblk; ++b) {
      auto rg = Range(0, w.size()).Segment(b, nblk);
      auto data = transposed.GetBlock().Slice(rg.begin, rg.end);
      data.label = rowblk.GetBlock().label;
      auto grad_seg = grad.segment(rg.begin, rg.end);
      auto param = {SArray<char>(pred), {}, SArray<char>(w.segment(rg.begin, rg.end))};
      loss.CalcGrad(data, param, &grad_seg);
    }

    EXPECT_LE(fabs(norm2(pred) - norm2(ref_pred)) / norm2(ref_pred), 1e-6);
    EXPECT_LE(fabs(norm2(grad) - norm2(ref_grad)) / norm2(ref_grad), 1e-6);
  }
}

TEST(LogitLossDelta, Hessien) {
  // load and tranpose data
  dmlc::data::RowBlockContainer<unsigned> rowblk, transposed;
  std::vector<feaid_t> uidx;
  load_data(&rowblk, &uidx);
  SpMT::Transpose(rowblk.GetBlock(), &transposed, uidx.size());

  // init loss
  KWArgs args = {{"compute_hession", "1"}};
  LogitLossDelta loss; loss.Init(args);

  // init weight
  SArray<real_t> weight(47149);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = i / 5e4;
  }
  SArray<real_t> w(uidx.size());
  for (size_t i = 0; i < uidx.size(); ++i) {
    w[i] = weight[uidx[i]];
  }


  int nblk = 10;
  SArray<real_t> pred(100), grad(w.size()*2);

  for (int b = 0; b < nblk; ++b) {
    auto rg = Range(0, w.size()).Segment(b, nblk);
    auto data = transposed.GetBlock().Slice(rg.begin, rg.end);
    data.label = rowblk.GetBlock().label;
    loss.Predict(data, {SArray<char>(w.segment(rg.begin, rg.end))}, &pred);
  }

  for (int b = 0; b < nblk; ++b) {
    auto rg = Range(0, w.size()).Segment(b, nblk);
    auto data = transposed.GetBlock().Slice(rg.begin, rg.end);
    data.label = rowblk.GetBlock().label;
    auto w_seg = w.segment(rg.begin, rg.end);
    auto grad_seg = grad.segment(rg.begin*2, rg.end*2);
    SArray<int> grad_pos(w_seg.size());
    for (size_t i = 0; i < w_seg.size(); ++i) grad_pos[i] = 2*i;
    auto param = {SArray<char>(pred), SArray<char>(grad_pos), SArray<char>(w_seg)};
    loss.CalcGrad(data, param, &grad_seg);
  }

  SArray<real_t> H(w.size()), G(w.size());
  for (size_t i = 0; i < w.size(); ++i) {
    G[i] = grad[i*2];
    H[i] = grad[i*2+1];
  }

  EXPECT_LT(fabs(norm2(G) - 90.5817), 1e-4);
  EXPECT_LT(fabs(norm2(H) - 0.0424518), 1e-6);
}
