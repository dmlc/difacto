#pragma once
#include "dmlc/data.h"
namespace difacto {

/**
 * \brief the basic class of a loss function
 */
class Loss {
 public:
  Loss() { }
  virtual ~Loss() { }

  void Init()
  /**
   * \brief create and init the loss function
   *
   * @param data X and Y
   * @param model the model entries
   * @param model_siz the i-th element is the length of the i-th model entry
   * @param conf config
   */
 void Init(const RowBlock<unsigned>& data,
           const std::vector<real_t>& model,
           const std::vector<int>& model_siz,
           const Config& conf) virtual = 0;

  /**
   * \brief evaluate the progress
   * \param prog the output progress
   */
  void Evaluate(std::vector<real_t>* prog) virtual = 0;

  /*!
   * \brief compute the gradients
   * \param grad the output gradients, often should have the same size as \ref model
   */
  void CalcGrad(std::vector<real_t>* grad) virtual = 0;

  /**
   * \brief predict
   * \param pred the prediction results, whose length should be equal to the
   * number of examples in \ref data
   */
  void Predict(std::vector<real_t>* pred) virtual = 0;

  /**
   * \brief factory function
   * \param type the loss type such as "fm"
   */
  Loss* Create(const std::string& type);
};

}  // namespace difacto
