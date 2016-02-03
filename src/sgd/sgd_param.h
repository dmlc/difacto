#ifndef _SGD_PARAM_H_
#define _SGD_PARAM_H_
#include "dmlc/parameter.h"
namespace difacto {
/**
 * \brief sgd config
 */
struct SGDLearnerParam : public dmlc::Parameter<SGDLearnerParam> {
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /**
   * \brief The optional validation dataset for a training task, either a
   *  filename or a directory
   */
  std::string val_data;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the model output for a training task */
  std::string model_out;
  /**
   * \brief the model input
   * should be specified if it is a prediction task, or a training
   */
  std::string model_in;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes */
  int max_num_epochs;
  /**
   * \brief the minibatch size
   */
  int batch_size;
  int shuffle;
  float neg_sampling;

  int job_size;

  DMLC_DECLARE_PARAMETER(SGDLearnerParam) {
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(val_data).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
    DMLC_DECLARE_FIELD(job_size).set_default(4);
  }
};
}  // namespace difacto
#endif  // _SGD_PARAM_H_
