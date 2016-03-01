/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_SGD_SGD_PARAM_H_
#define DIFACTO_SGD_SGD_PARAM_H_
#include <string>
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
  std::string data_val;
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

  /** \brief issue num_jobs_per_epoch * num_workers per epoch */
  int num_jobs_per_epoch;

  /** \brief show the training progress for every n second */
  int report_interval;
  /** \brief stop if (objv_new - objv_old) / obj_old < threshold */
  real_t stop_rel_objv;
  /** \brief stop if val_auc_new - val_auc_old < threshold */
  real_t stop_val_auc;
  DMLC_DECLARE_PARAMETER(SGDLearnerParam) {
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_val).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
    DMLC_DECLARE_FIELD(num_jobs_per_epoch).set_default(10);
    DMLC_DECLARE_FIELD(batch_size);
    DMLC_DECLARE_FIELD(shuffle).set_default(10);
    DMLC_DECLARE_FIELD(neg_sampling).set_default(1);
    DMLC_DECLARE_FIELD(stop_rel_objv).set_default(1e-5);
    DMLC_DECLARE_FIELD(stop_val_auc).set_default(1e-5);
  }
};

struct SGDUpdaterParam : public dmlc::Parameter<SGDUpdaterParam> {
  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;
  /** \brief the l2 regularizer for :math:`V`: :math:`\lambda_2 \|V_i\|_2^2` */
  float V_l2;

  /** \brief the learning rate :math:`\eta` (or :math:`\alpha`) for :math:`w` */
  float lr;
  /** \brief learning rate :math:`\beta` */
  float lr_beta;
  /** \brief learning rate :math:`\eta` for :math:`V`. */
  float V_lr;
  /** \brief leanring rate :math:`\beta` for :math:`V`. */
  float V_lr_beta;
  /**
   * \brief the scale to initialize V.
   * namely V is initialized by uniform random number in
   *   [-V_init_scale, +V_init_scale]
   */
  float V_init_scale;
  /** \brief the embedding dimension */
  int V_dim;
  /** \brief the minimal feature count for allocating V */
  int V_threshold;
  /** \brief random seed */
  unsigned int seed;
  DMLC_DECLARE_PARAMETER(SGDUpdaterParam) {
    DMLC_DECLARE_FIELD(l1).set_range(0, 1e10).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_range(0, 1e10).set_default(0);
    DMLC_DECLARE_FIELD(V_l2).set_range(0, 1e10).set_default(.01);
    DMLC_DECLARE_FIELD(lr).set_range(0, 10).set_default(.01);
    DMLC_DECLARE_FIELD(lr_beta).set_range(0, 1e10).set_default(1);
    DMLC_DECLARE_FIELD(V_lr).set_range(0, 1e10).set_default(.01);
    DMLC_DECLARE_FIELD(V_lr_beta).set_range(0, 10).set_default(1);
    DMLC_DECLARE_FIELD(V_init_scale).set_range(0, 10).set_default(.01);
    DMLC_DECLARE_FIELD(V_threshold).set_default(10);
    DMLC_DECLARE_FIELD(V_dim);
    DMLC_DECLARE_FIELD(seed).set_default(0);
  }
};
}  // namespace difacto
#endif  // DIFACTO_SGD_SGD_PARAM_H_
