/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "lbfgs/lbfgs_learner.h"

using namespace difacto;

TEST(LBFGSLearner, Basic) {
  std::vector<real_t> objv = {
    34.603421,
    12.655075,
    5.224232,
    2.713903,
    1.290586,
    0.645131,
    0.317889,
    0.156723,
    0.075331,
    0.032091,
    0.018044,
    0.008562,
    0.004336,
    0.002132,
    0.001051,
    0.000506,
    0.000227,
    0.000119,
    0.000059};

  LBFGSLearner learner;
  KWArgs args = {{"data_in", "../tests/data"},
                 {"m", "5"},
                 {"V_dim", "0"},
                 {"tail_feature_filter", "0"},
                 {"max_num_epochs", "19"}};
  auto remain = learner.Init(args);
  EXPECT_EQ(remain.size(), 0);

  auto callback = [objv](int epoch, const lbfgs::Progress& prog) {
    EXPECT_LT(fabs(objv[epoch] - prog.objv), 1e-5);
  };
  learner.AddEpochEndCallback(callback);
  learner.Run();
}

TEST(LBFGSLearner, RemoveTailFeatures) {
  std::vector<real_t> objv = {
    43.865008,
    21.728511,
    10.893458,
    5.038567,
    2.293318,
    1.064151,
    0.518891,
    0.257997,
    0.128646,
    0.064974,
    0.028329,
    0.016543,
    0.007910,
    0.004053,
    0.002001,
    0.000978,
    0.000437,
    0.000216,
    0.000112};
  LBFGSLearner learner;
  KWArgs args = {{"data_in", "../tests/data"},
                 {"m", "5"},
                 {"V_dim", "0"},
                 {"tail_feature_filter", "2"},
                 {"max_num_epochs", "19"}};
  auto remain = learner.Init(args);
  EXPECT_EQ(remain.size(), 0);
  auto callback = [objv](int epoch, const lbfgs::Progress& prog) {
    EXPECT_LT(fabs(objv[epoch] - prog.objv), 1e-5);
  };
  learner.AddEpochEndCallback(callback);
  learner.Run();
}
