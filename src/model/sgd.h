/**
 * @file   sgd.h
 * @brief  the stochastic gradient descent solver
 */
#ifndef DIFACTO_MODEL_SGD_H_
#define DIFACTO_MODEL_SGD_H_
#include <string>
#include <vector>
#include <limits>
#include "difacto/model.h"
#include "dmlc/parameter.h"
#include "dmlc/io.h"
namespace difacto {

struct SGDModelParam : public dmlc::Parameter<SGDModelParam> {
  /** \brief the l1 regularizer for :math:`w`: :math:`\lambda_1 |w|_1` */
  float l1;
  /** \brief the l2 regularizer for :math:`w`: :math:`\lambda_2 \|w\|_2^2` */
  float l2;
  /** \brief the l2 regularizer for :math:`V`: :math:`\lambda_2 \|V_i\|_2^2` */
  float V_l2;
  /** \brief the embedding dimension */
  int V_dim;
  DMLC_DECLARE_PARAMETER(SGDModelParam) {
    DMLC_DECLARE_FIELD(l1).set_range(0, 1e10).set_default(1);
    DMLC_DECLARE_FIELD(l2).set_range(0, 1e10).set_default(0);
    DMLC_DECLARE_FIELD(V_l2).set_range(0, 1e10).set_default(.01);
    DMLC_DECLARE_FIELD(V_dim);
  }
};

struct SGDOptimizerParam : public dmlc::Parameter<SGDOptimizerParam> {
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

  int V_threshold;
  DMLC_DECLARE_PARAMETER(SGDOptimizerParam) {
    DMLC_DECLARE_FIELD(lr).set_range(0, 10).set_default(.01);
    DMLC_DECLARE_FIELD(lr_beta).set_range(0, 1e10).set_default(1);
    DMLC_DECLARE_FIELD(V_lr).set_range(0, 1e10).set_default(.01);
    DMLC_DECLARE_FIELD(V_lr_beta).set_range(0, 10).set_default(1);
    DMLC_DECLARE_FIELD(V_init_scale).set_range(0, 10).set_default(.01);
    DMLC_DECLARE_FIELD(V_dim);
  }
};

/**
 * \brief the weight entry for one feature
 */
struct SGDEntry {
 public:
  SGDEntry() { fea_cnt = 0; w = 0; sqrt_g = 0; z = 0; V = nullptr; }
  ~SGDEntry() { delete [] V; }

  /** \brief the number of appearence of this feature in the data so far */
  real_t fea_cnt;
  /** \brief w and its aux data */
  real_t w, sqr_g, z;
  /** \brief V and its aux data */
  real_t *V;
};

/**
 * \brief store all weights
 */
class SGDModel {
 public:
  SGDModel() { }
  ~SGDModel() { }
  /**
   * \brief init model
   *
   * @param start_id the minimal feature id
   * @param end_id the maximal feature id
   */
  KWArgs Init(const KWArgs& kwargs, feaid_t start_id, feaid_t end_id);
  /**
   * \brief get the weight entry for a feature id
   * \param id the feature id
   */
  inline SGDEntry& operator[] (feaid_t id) {
    CHECK_GE(id, start_id_);
    id -= start_id_;
    return dense_ ? model_vec_[id] : model_map_[id];
  }
  /**
   * \brief load model
   * \param fi input stream
   */
  void Load(dmlc::Stream* fi, bool* has_aux);
  /**
   * \brief save model
   * \param fo output stream
   */
  void Save(bool save_aux, dmlc::Stream *fo) const;

 private:
  /** \brief load one entry */
  inline void Load(dmlc::Stream* fi, int len, SGDEntry* entry);

  /** \brief save one entry */
  inline void Save(
      bool save_aux, feaid_t id, const SGDEntry& entry, dmlc::Stream *fo);

  bool dense_;
  feaid_t start_id_;
  std::vector<SGDEntry<T>> model_vec_;
  std::unordered_map<SGDEntry<T>> model_map_;
  SGDModelParam param_;
};

/**
 * \brief sgd optimizer
 *
 * - w is updated by FTRL, which is a smooth version of adagrad works well with
 *   the l1 regularizer
 * - V is updated by adagrad
 */
class SGDOptimizer : public Model {
 public:
  SGD() { }
  virtual ~SGD() { }

  KWArgs Init(const KWArgs& kwargs) override {
    auto remain = param_.InitAllowUnknown(KWArgs);
    remain.push_back("V_dim", std::to_string(param_.V_dim));
    remain = model_.Init(remain, 0, std::numeric_limits<feaid_t>::max());
    return remain;
  }

  void Load(dmlc::Stream* fi, bool* has_aux) override {
    model_.Load(fi, has_aux);
  }

  void Save(bool save_aux, dmlc::Stream *fo) const override {
    model_.Save(save_aux, fo);
  }

  void Get(const std::vector<feaid_t>& fea_ids,
           std::vector<T>* weights,
           std::vector<int>* weight_lens) override;

  void AddCount(const std::vector<feaid_t>& fea_ids,
                const std::vector<uint32_t>& fea_cnts) override;

  void Update(const std::vector<ps::Key>& fea_ids,
              const std::vector<T>& grad,
              const std::vector<int>& grad_lens) override;


 private:
  /** \brief update w by FTRL */
  void UpdateW(real_t gw, SGDEntry* e);

  /** \brief update V by adagrad */
  void UpdateV(real_t const* gV, SGDEntry* e);

  /** \brief init V */
  void InitV(SGDEntry* e);

  SGDModel model_;
  SGDOptimizerParam param_;

  int64_t new_w_;
};


}  // namespace difacto
#endif  // DIFACTO_MODEL_SGD_H_
