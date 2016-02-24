/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "sgd/sgd_learner.h"

using namespace difacto;

TEST(SGDLearner, Basic) {
  std::vector<real_t> objv = {
    69.314718,
    69.314718,
    67.151912,
    61.414778,
    56.244989,
    53.218700,
    51.248737,
    49.846688,
    48.650164,
    47.698351,
    46.924038,
    46.388223,
    45.970721,
    45.499307,
    45.102245,
    44.798413,
    44.565211,
    44.386417,
    44.240657,
    44.109764};
  SGDLearner learner;
  KWArgs args = {{"data_in", "../tests/data"},
                 {"V_dim", "0"},
                 {"l2", "1"},
                 {"l1", "1"},
                 {"lr", "1"},
                 {"num_jobs_per_epoch", "1"},
                 {"batch_size", "100"},
                 {"max_num_epochs", "20"}};
  auto remain = learner.Init(args);
  EXPECT_EQ(remain.size(), 0);

  auto callback = [objv](
      int epoch, const sgd::Progress& train, const sgd::Progress& val) {
    EXPECT_LT(fabs(objv[epoch] - train.loss), 5e-5);
  };
  learner.AddEpochEndCallback(callback);
  learner.Run();
}
