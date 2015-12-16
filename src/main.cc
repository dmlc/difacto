#include "difacto/difacto.h"
#include "learner/sgd.h"
#include "common/arg_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "usage: difacto key1=val1 key2=val2 ...";
    return 0;
  }

  using namespace difacto;
  ArgParser parser;
  for (int i = 1; i < argc; ++i) {
    parser.AddArg(argv[i]);
  }

  DiFacto df;
  df.Init(parser.GetKWArgs());
  df.Run();
  return 0;
}
