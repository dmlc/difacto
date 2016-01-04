#include <gtest/gtest.h>
#include "data/batch_iter.h"
#include "./utils.h"
#include "loss/fm_loss.h"
#include "common/localizer.h"
#include "loss/bin_class_eval.h"

using namespace difacto;

TEST(FMLoss, NoV) {
  SArray<real_t> weight(47149);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = i / 5e4;
    // weight[i] = 1;
  }

  BatchIter iter("../tests/data", "libsvm", 0, 1, 100);
  CHECK(iter.Next());
  dmlc::data::RowBlockContainer<unsigned> compact;
  std::vector<feaid_t> uidx;
  Localizer lc;
  lc.Compact(iter.Value(), &compact, &uidx);

  SArray<real_t> compact_w(uidx.size());
  for (size_t i = 0; i < uidx.size(); ++i) {
    compact_w[i] = weight[ReverseBytes(uidx[i])];
  }

  KWArgs args = {{"V_dim", "0"}};
  FMLoss loss;
  loss.Init(args);
  SArray<real_t> pred;
  auto data = compact.GetBlock();
  loss.Predict(data,
               {SArray<char>(compact_w), SArray<char>()},
               &pred);

  BinClassEval eval(data.label, pred.data(), data.size);

  // Progress prog;
  EXPECT_LT(fabs(eval.LogitObjv() - 147.4672), 1e-3);

  SArray<real_t> grad;
  loss.CalcGrad(data, {SArray<char>(compact_w), SArray<char>(),
          SArray<char>(pred)}, &grad);
  EXPECT_LT(fabs(norm2(grad) - 90.5817), 1e-3);
}


TEST(FMLoss, HasV) {
  int V_dim = 5;
  int n = 47149;
  std::vector<real_t> w(n*(V_dim+1));
  for (int i = 0; i < n; ++i) {
    w[i*(V_dim+1)] = i / 5e4;
    for (int j = 1; j <= V_dim; ++j) {
      w[i*(V_dim+1)+j] = i * j / 5e5;
    }
  }

  BatchIter iter("../tests/data", "libsvm", 0, 1, 100);
  CHECK(iter.Next());
  dmlc::data::RowBlockContainer<unsigned> compact;
  std::vector<feaid_t> uidx;
  Localizer lc;
  lc.Compact(iter.Value(), &compact, &uidx);

  SArray<int> len(uidx.size());
  SArray<real_t> compact_w(uidx.size()*(V_dim+1));
  for (size_t i = 0; i < uidx.size(); ++i) {
    for (int j = 0; j < V_dim+1; ++j) {
      compact_w[i*(V_dim+1)+j] = w[ReverseBytes(uidx[i])*(V_dim+1)+j];
    }
    len[i] = V_dim+1;
  }

  KWArgs args = {{"V_dim", std::to_string(V_dim)}};
  FMLoss loss;
  loss.Init(args);
  auto data = compact.GetBlock();
  SArray<real_t> pred;
  loss.Predict(data, {SArray<char>(compact_w), SArray<char>(len)}, &pred);

  // Progress prog;
  BinClassEval eval(data.label, pred.data(), data.size);
  EXPECT_LT(fabs(eval.LogitObjv() - 330.628), 1e-3);

  SArray<real_t> grad(compact_w.size());
  loss.CalcGrad(data, {SArray<char>(compact_w), SArray<char>(len),
          SArray<char>(pred)}, &grad);
  EXPECT_LT(fabs(norm2(grad) - 1.2378e+03), 1e-1);
}
