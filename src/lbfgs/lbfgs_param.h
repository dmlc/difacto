/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_LBFGS_LBFGS_PARAM_H_
#define DIFACTO_LBFGS_LBFGS_PARAM_H_
#include <string>
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
  /** \brief the maximal number of data passes, defaut is 100 */
  int max_num_epochs;
  /** \brief the minimal number of data passes, defaut is 10 */
  int min_num_epochs;
  /** \brief the size of data in MB read each time for processing, in default 256 MB */
  real_t data_chunk_size;

  /** \brief stop if (objv_new - objv_old) / obj_old < threshold */
  real_t stop_rel_objv;

  /** \brief stop if val_auc_new - val_auc_old < threshold */
  real_t stop_val_auc;
  int load_epoch;

  /** \brief starting alpha for epoch 0, in default determined by system */
  real_t init_alpha;
  /** \brief starting alpha for each epoch (except 0), in default is 1 */
  real_t alpha;
  real_t c1;
  real_t c2;
  real_t rho;
  int max_num_linesearchs;

  int num_threads;

  DMLC_DECLARE_PARAMETER(LBFGSLearnerParam) {
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_val).set_default("");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_cache).set_default("/tmp/difacto_lbfgs_");
    DMLC_DECLARE_FIELD(data_chunk_size).set_default(256);
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(100);
    DMLC_DECLARE_FIELD(min_num_epochs).set_default(10);
    DMLC_DECLARE_FIELD(alpha).set_default(1);
    DMLC_DECLARE_FIELD(init_alpha).set_default(0);
    DMLC_DECLARE_FIELD(max_num_linesearchs).set_default(5);
    DMLC_DECLARE_FIELD(c1).set_default(1e-4);
    DMLC_DECLARE_FIELD(c2).set_default(.9);
    DMLC_DECLARE_FIELD(rho).set_default(.5);
    DMLC_DECLARE_FIELD(load_epoch).set_default(0);
    DMLC_DECLARE_FIELD(stop_rel_objv).set_default(1e-5);
    DMLC_DECLARE_FIELD(stop_val_auc).set_default(1e-4);
    DMLC_DECLARE_FIELD(num_threads).set_default(0);
  }
};

}  // namespace difacto
#endif  // DIFACTO_LBFGS_LBFGS_PARAM_H_
