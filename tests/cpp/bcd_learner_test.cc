#include <gtest/gtest.h>
#include "bcd/bcd_learner.h"

using namespace difacto;

TEST(BCDLearer, DiagNewton) {
  BCDLearner learner;
  KWArgs args = {{"task", "train"},
                 {"data_in", "../tests/data"},
                 {"l1", ".1"},
                 {"lr", ".05"},
                 {"block_ratio", "0.001"},
                 {"tail_feature_filter", "0"},
                 {"max_num_epochs", "10"}};
  auto remain = learner.Init(args);
  EXPECT_EQ(remain.size(), 0);

  std::vector<real_t> objv = {
    34.877064,
    33.885559,
    29.572740,
    27.458964,
    25.317689,
    23.917098,
    22.855843,
    22.099876,
    21.552682,
    21.137216
  };

  auto callback = [objv](int epoch, const bcd::Progress& prog) {
    EXPECT_LT(abs(prog.value[0] - objv[epoch]), 1e-5);
  };
  learner.AddEpochEndCallback(callback);
  learner.Run();
}

// the optimal solution with ../tests/data and l1 = .1 is objv = 15.884923, nnz
// w = 47

TEST(BCDLearer, Convergence) {

  std::vector<real_t> ratio = {.4, 1, 10};

  for (real_t r : ratio) {
    real_t objv;
    BCDLearner learner;
    KWArgs args = {{"task", "train"},
                   {"data_in", "../tests/data"},
                   {"l1", ".1"},
                   {"lr", ".8"},
                   {"block_ratio", std::to_string(r)},
                   {"tail_feature_filter", "0"},
                   {"max_num_epochs", "100"}};
    auto remain = learner.Init(args);
    EXPECT_EQ(remain.size(), 0);

    auto callback = [&objv](int epoch, const bcd::Progress& prog) {
      objv = prog.value[0];
    };
    learner.AddEpochEndCallback(callback);
    learner.Run();

    EXPECT_LT(abs(objv - 15.884923), .001);
  }

}
