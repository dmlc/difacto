#include "./utils.h"
#include "common/arg_parser.h"
#include "common/spmv.h"
#include "dmlc/config.h"
#include "dmlc/timer.h"
#include "reader/reader.h"

using namespace difacto;
using namespace dmlc;

struct Param : public Parameter<Param> {
  std::string data;
  std::string format;
  int nthreads;
  DMLC_DECLARE_PARAMETER(Param) {
    DMLC_DECLARE_FIELD(format).set_default("libsvm").describe("data format");;
    DMLC_DECLARE_FIELD(data).describe("input data filename");;
    DMLC_DECLARE_FIELD(nthreads).set_default(2).describe("number of threads");;
  }
};

DMLC_REGISTER_PARAMETER(Param);

int main(int argc, char *argv[]) {
  Param param;
  if (argc < 2) {
    LOG(ERROR) << "not enough input.. \n\nusage: ./difacto key1=val1 key2=val2 ...\n\n" << param.__DOC__();
    return 0;
  }
  ArgParser parser;
  for (int i = 1; i < argc; ++i) parser.AddArg(argv[i]);
  param.Init(parser.GetKWArgs());

  Reader reader(param.data, param.format, 0, 1, 512<<20);
  CHECK(reader.Next());
  dmlc::data::RowBlockContainer<unsigned> data;
  std::vector<feaid_t> uidx;
  Localizer lc; lc.Compact(reader.Value(), &data, &uidx);
  auto D = data.GetBlock();
  size_t n = D.size;
  size_t p = uidx.size();
  LOG(INFO) << "load " << n << " x " << p << " matrix";

  double start;
  int repeat = 20;
  SArray<real_t> x(p), y(n);

  for (int i = 0; i < repeat+1; ++i) {
    if (i == 1) start = GetTime();  // warmup when i == 0
    SpMV::Times(D, x, &y, param.nthreads);
  }
  double t1 = (GetTime() - start) / repeat;

  for (int i = 0; i < repeat+1; ++i) {
    if (i == 1) start = GetTime();  // warmup when i == 0
    SpMV::TransTimes(D, y, &x, param.nthreads);
  }
  double t2 = (GetTime() - start) / repeat;


  LOG(INFO) << "Times: " << t1 << ",\t TransTimes: " << t2;

  return 0;
}
