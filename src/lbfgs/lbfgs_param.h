#ifndef _LBFGS_PARAM_H_
#define _LBFGS_PARAM_H_
#include "dmlc/parameter.h"
namespace difacto {

struct LBFGSLearnerParam : public dmlc::Parameter<LBFGSLearnerParam> {
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /** \brief The optional validation dataset, either a filename or a directory */
  std::string data_val;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the directory for the data chache */
  std::string data_cache;
  /** \brief the model output */
  std::string model_out;
  /** \brief the model input for warm start */
  std::string model_in;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes, defaut is 20 */
  int max_num_epochs;
  /** \brief the size of data in MB read each time for processing, in default 256 MB */
  int data_chunk_size;

  int load_epoch;

  real_t alpha;
  real_t c1;
  real_t c2;
  real_t rho;

  DMLC_DECLARE_PARAMETER(LBFGSLearnerParam) {
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_val).set_default("");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_cache).set_default("/tmp/difacto_lbfgs_");
    DMLC_DECLARE_FIELD(data_chunk_size).set_default(1<<28);
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
  }
};

}  // namespace difacto
#endif  // _LBFGS_PARAM_H_
