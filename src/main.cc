/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "common/arg_parser.h"
#include "dmlc/parameter.h"
#include "reader/converter.h"
namespace difacto {
struct DifactoParam : public dmlc::Parameter<DifactoParam> {
  /**
   * \brief the type of task,
   * - train: train a model, which is the default
   * - predict: predict by using a trained model
   * - convert: convert data from one format into another
   */
  std::string task;
  /** \brief the learner's type, required for a training task */
  std::string learner;
  DMLC_DECLARE_PARAMETER(DifactoParam) {
    DMLC_DECLARE_FIELD(learner).set_default("sgd");
    DMLC_DECLARE_FIELD(task).set_default("train");
  }
};

void WarnUnknownKWArgs(const DifactoParam& param, const KWArgs& remain) {
  if (remain.empty()) return;
  LOG(WARNING) << "unrecognized keyword argument for task " << param.task;
  for (auto kw : remain) {
    LOG(WARNING) << "  " << kw.first << " = " << kw.second;
  }
}

DMLC_REGISTER_PARAMETER(DifactoParam);
DMLC_REGISTER_PARAMETER(ConverterParam);

}  // namespace difacto

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto key1=val1 key2=val2 ...";
    return 0;
  }
  using namespace difacto;

  // parse configuure
  ArgParser parser;
  for (int i = 1; i < argc; ++i) parser.AddArg(argv[i]);
  DifactoParam param;
  auto kwargs_remain = param.InitAllowUnknown(parser.GetKWArgs());

  // run
  if (param.task == "train") {
    Learner* learner = Learner::Create(param.learner);
    WarnUnknownKWArgs(param, learner->Init(kwargs_remain));
    learner->Run();
    delete learner;
  } else if (param.task == "convert") {
    Converter converter;
    WarnUnknownKWArgs(param, converter.Init(kwargs_remain));
    converter.Run();
  } else if (param.task == "predict") {
    LOG(FATAL) << "TODO";
  } else {
    LOG(FATAL) << "unknown task: " << param.task;
  }
  return 0;
}
