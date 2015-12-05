#include "difacto/difacto.h"
#include "learner/sgd.h"
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

  DiFacto df;
  df.Init(parser.GetKWArgs());
  df.Run();
  return 0;
}
