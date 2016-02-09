/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_BCD_BCD_PARAM_H_
#define DIFACTO_BCD_BCD_PARAM_H_
#include "dmlc/parameter.h"
namespace difacto {

struct BCDLearnerParam : public dmlc::Parameter<BCDLearnerParam> {
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /** \brief The optional validation dataset, either a filename or a directory */
  std::string data_val;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the directory for the data chache */
  std::string data_cache;
  /** \brief the model output for a training task */
  std::string model_out;
  /** \brief the model input for warm start */
  std::string model_in;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes, defaut is 20 */
  int max_num_epochs;
  /** \brief controls the number of feature blocks, default is 4 */
  float block_ratio;
  /** \brief if or not process feature blocks in a random order, default is true */
  int random_block;
  /** \brief the number of heading bits used to encode the feature group, default is 12 */
  int num_feature_group_bits;
  float neg_sampling;
  /** \brief the size of data in MB read each time for processing, in default 256 MB */
  int data_chunk_size;

  DMLC_DECLARE_PARAMETER(BCDLearnerParam) {
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_val).set_default("");
    DMLC_DECLARE_FIELD(data_cache).set_default("/tmp/difacto_bcd_");
    DMLC_DECLARE_FIELD(data_chunk_size).set_default(1<<28);
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
    DMLC_DECLARE_FIELD(random_block).set_default(1);
    DMLC_DECLARE_FIELD(num_feature_group_bits).set_default(0);
    DMLC_DECLARE_FIELD(block_ratio).set_default(4);
  }
};
}  // namespace difacto
#endif  // DIFACTO_BCD_BCD_PARAM_H_
