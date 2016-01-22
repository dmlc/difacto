/**
 * Copyright (c) 2015 by Contributors
 */
#include "difacto/learner.h"
#include "common/arg_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto key1=val1 key2=val2 ...";
    return 0;
  }

  using namespace difacto;
  ArgParser parser;
  for (int i = 1; i < argc; ++i) parser.AddArg(argv[i]);

  Learner* learner = Learner::Create("xxx");
  auto remain = learner->Init(parser.GetKWArgs());
  if (!remain.empty()) {
    LOG(WARNING) << "unrecognized keyword argument:";
    for (auto kw : remain) LOG(WARNING) << "  " << kw.first << " = " << kw.second;
  }
  learner->Run();
  delete learner;
  return 0;
}
