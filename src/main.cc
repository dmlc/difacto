#include "difacto/difacto.h"
#include "model/sgd.h"
#include "common/arg_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto arg_file [args]";
    LOG(ERROR) << " arg_file: use 'none' if no arg_file";
    return 0;
  }

  using namespace difacto;
  ArgParser parser;
  if (strcmp(argv[1], "none")) {
    parser.AddArgFile(argv[1]);
  }
  for (int i = 2; i < argc; ++i) {
    parser.AddArg(argv[i]);
  }

  SGDModelParam param_;
  param_.Init(parser.GetKWArgs());

  LOG(ERROR) << param_.l2;
  LOG(ERROR) << param_.l1;

  // DiFacto df;
  // df.Init({});
  // df.Run();
  return 0;
}
