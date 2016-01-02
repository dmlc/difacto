#include <gtest/gtest.h>
#include "data/batch_iter.h"
#include "./utils.h"
#include "loss/fm_loss.h"
#include "common/localizer.h"

using namespace difacto;

TEST(FMLoss, NoV) {
  std::vector<real_t> weight(47149);
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

  std::vector<real_t> compact_w(uidx.size());
  for (size_t i = 0; i < uidx.size(); ++i) {
    compact_w[i] = weight[ReverseBytes(uidx[i])];
  }

  KWArgs args = {{"V_dim", "0"}};
  FMLoss loss;
  loss.Init(args);
  loss.InitData(compact.GetBlock(), compact_w, std::vector<int>());

  // Progress prog;
  // loss.Evaluate(&prog);
  // EXPECT_LT(fabs(prog.objv() - 147.4672), 1e-3);

  std::vector<real_t> grad;
  loss.CalcGrad(&grad);
  EXPECT_LT(fabs(norm2(grad.data(), grad.size()) - 90.5817), 1e-3);
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

  std::vector<int> len(uidx.size());
  std::vector<real_t> compact_w(uidx.size()*(V_dim+1));
  for (size_t i = 0; i < uidx.size(); ++i) {
    for (int j = 0; j < V_dim+1; ++j) {
      compact_w[i*(V_dim+1)+j] = w[ReverseBytes(uidx[i])*(V_dim+1)+j];
    }
    len[i] = V_dim+1;
  }

  KWArgs args = {{"V_dim", std::to_string(V_dim)}};
  FMLoss loss;
  loss.Init(args);
  loss.InitData(compact.GetBlock(), compact_w, len);

  // Progress prog;
  // loss.Evaluate(&prog);
  // EXPECT_LT(fabs(prog.objv() - 330.628), 1e-3);

  std::vector<real_t> grad(compact_w.size());
  loss.CalcGrad(&grad);
  EXPECT_LT(fabs(norm2(grad.data(), grad.size()) - 1.2378e+03), 1e-1);
}
