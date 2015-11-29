#pragma once

namespace difacto {

/**
 * \brief the commnunication interface of a model storage, which is used by
 * workers to get / set the model
 * \tparam T the gradient/weight data type
 */
template <typename T>
class ModelSync {
 public:
  ModelStore() { }
  virtual ~ModelStore() { }

  /**
   * \brief use shared pointer for communication
   */
  template <typename V>
  using SharedVector = std::shared_ptr<std::vector<V>>;

  virtual void Push(const SharedVector<feaid_t>& fea_ids,
                    const SharedVector<int>& cnt) = 0;

  virtual void Push(const SharedVector<feaid_t>& fea_ids,
                    const SharedVector<T>& vals,
                    const SharedVector<int>& lens) = 0;

  virtual void Pull(const SharedVector<feaid_t>& fea_ids,
                    std::vector<T>* vals,
                    std::vector<int>* lens) = 0;


  static ModelStore<T>* Create(const std::string& type);
};

}  // namespace difacto
