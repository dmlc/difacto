
#include <gtest/gtest.h>
#include "sgd/sgd_learner.h"

using namespace difacto;

TEST(SGDLearner, Basic) {
  SGDLearner learner;
  KWArgs args = {{"data_in", "../tests/data"},
                 {"V_dim", "0"},
                 {"l2", "0"},
                 {"num_jobs_per_epoch", "1"},
                 {"batch_size", "100"},
                 {"max_num_epochs", "19"}};
  auto remain = learner.Init(args);
  EXPECT_EQ(remain.size(), 0);

  // auto callback = [objv](int epoch, const lbfgs::Progress& prog) {
  //   EXPECT_LT(fabs(objv[epoch] - prog.objv), 1e-5);
  // };
  // learner.AddEpochEndCallback(callback);
  learner.Run();
}
