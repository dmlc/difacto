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
  for (int i = 1; i < argc; ++i) {
    parser.AddArg(argv[i]);
  }

  Learner* learner = Learner::Create("xxx");
  learner->Init(parser.GetKWArgs());
  learner->Run();
  delete learner;
  return 0;
}


    // DMLC_DECLARE_FIELD(argfile).set_default("");

  /** \brief filename for kwargs */
  // std::string argfile;
  /** \brief number of threads */
//   int num_threads;


//     DMLC_DECLARE_FIELD(num_threads).set_default(2);

//   void RunScheduler();
//   /**
//    * \brief schedule the jobs for one epoch
//    */
//   void RunEpoch(int epoch, int job_type);
//   /**
//    * \brief process a job
//    */
//   void Process(const Job& job);
//   /**
//    * \brief process a training/prediction job
//    */
//   void ProcessFile(const Job& job);

//   /**
//    * \brief sleep for a moment
//    * \param ms the milliseconds (1e-3 sec) for sleeping
//    */
//   inline void Sleep(int ms = 1000) {
//     std::this_thread::sleep_for(std::chrono::milliseconds(ms));
//   }

//   Progress progress_;
//   ProgressPrinter pprinter_;
//   double worktime_;

//   // dmlc::Stream pred_out_;
// };
