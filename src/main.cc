/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "common/arg_parser.h"
#include "dmlc/parameter.h"

struct DifactoParam : public dmlc::Parameter<DifactoParam> {
  /** \brief the learner's type */
  std::string learner;
  DMLC_DECLARE_PARAMETER(DifactoParam) {
    DMLC_DECLARE_FIELD(learner).set_default("sgd");
  }
};
DMLC_REGISTER_PARAMETER(DifactoParam);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto key1=val1 key2=val2 ...";
    return 0;
  }
  using namespace difacto;

  // parse configu
  ArgParser parser;
  for (int i = 1; i < argc; ++i) parser.AddArg(argv[i]);
  DifactoParam param;
  auto kwargs_remain = param.InitAllowUnknown(parser.GetKWArgs());

  // init learner
  Learner* learner = Learner::Create(param.learner);
  kwargs_remain = learner->Init(kwargs_remain);
  if (!kwargs_remain.empty()) {
    LOG(WARNING) << "unrecognized keyword argument:";
    for (auto kw : kwargs_remain) {
      LOG(WARNING) << "  " << kw.first << " = " << kw.second;
    }
  }

  // run
  learner->Run();
  delete learner;
  return 0;
}
