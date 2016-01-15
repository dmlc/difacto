#include <gtest/gtest.h>
#include "loss/logit_loss_delta.h"
#include "common/localizer.h"
#include "loss/bin_class_eval.h"
#include "difacto/sarray.h"
#include "loss/logit_loss_delta.h"

using namespace difacto;

// void comp_grad(RowBlock<unsigned> data,
//                SArray<real_t> weight,
//                LogitLossDelta loss) {

// }

TEST(LogitLossDelt, Basic) {
  // SArray<real_t> weight(47149);
  // for (size_t i = 0; i < weight.size(); ++i) {
  //   weight[i] = i / 5e4;
  // }

  // dmlc::data::RowBlockContainer<unsigned> rowblk, transposed;
  // std::vector<feaid_t> uidx;
  // load_data(&rowblk, &uidx);
  // SpMT::Transpose(rowblk->GetBlock(), &transposed, udix.size());


  // SArray<real_t> w(uidx.size());
  // for (size_t i = 0; i < uidx.size(); ++i) {
  //   w[i] = weight[uidx[i]];
  // }

  // KWArgs args = {{"num_threads", "2"}};
  // LogitLossDelta loss; loss.Init(args);

  // SArray<real_t> pred;
  // auto data = rowblk.GetBlock();
  // loss.Predict(data,
  //              {SArray<char>(w), SArray<char>()},
  //              &pred);

  // BinClassEval eval(data.label, pred.data(), data.size);

  // // Progress prog;
  // EXPECT_LT(fabs(eval.LogitObjv() - 147.4672), 1e-3);

  // SArray<real_t> grad;
  // loss.CalcGrad(data, {SArray<char>(w), SArray<char>(),
  //         SArray<char>(pred)}, &grad);
  // EXPECT_LT(fabs(norm2(grad) - 90.5817), 1e-3);

  // BatchIter iter("../tests/data", "libsvm", 0, 1, 100);
  // CHECK(iter.Next());
  // dmlc::data::RowBlockContainer<unsigned> compact;
  // std::vector<feaid_t> uidx;
  // Localizer lc;
  // lc.Compact(iter.Value(), &compact, &uidx);

  // SArray<real_t> compact_w(uidx.size());
  // for (size_t i = 0; i < uidx.size(); ++i) {
  //   compact_w[i] = weight[ReverseBytes(uidx[i])];
  // }

  // LogitLossDelta loss;
  // SArray<real_t> pred;
  // auto data = compact.GetBlock();
  // loss.Predict(data,
  //              {SArray<char>(compact_w), SArray<char>()},
  //              &pred);

  // BinClassEval eval(data.label, pred.data(), data.size);

  // // Progress prog;
  // EXPECT_LT(fabs(eval.LogitObjv() - 147.4672), 1e-3);

  // SArray<real_t> grad;
  // loss.CalcGrad(data, {SArray<char>(compact_w), SArray<char>(),
  //         SArray<char>(pred)}, &grad);
  // EXPECT_LT(fabs(norm2(grad) - 90.5817), 1e-3);
}
