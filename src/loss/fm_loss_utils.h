#ifndef _FM_LOSS_UTILS_H_
#define _FM_LOSS_UTILS_H_
#include "difacto/sarray.h"
namespace difacto {
namespace fmloss {

/**
 * \brief internal data for the linear term of FM loss
 */
struct Linear {
  void Init(const SArray<real_t>& model,
            const SArray<int>& model_siz) {
    CHECK(!inited_);
    if (model_siz.empty()) {
      value.resize(model.size());
      memcpy(value.data(), model.data(), model.size()*sizeof(real_t));
    } else {
      pos.resize(model_siz.size());
      value.resize(model_siz.size());
      unsigned p = 0;
      for (size_t i = 0; i < model_siz.size(); ++i) {
        if (model_siz[i] == 0) {
          pos[i] = (unsigned)-1;
        } else {
          pos[i] = p; value[i] = model[p]; p += model_siz[i];
        }
      }
      CHECK_EQ((size_t)p, model.size());
    }
    inited_ = true;
  }

  /**
   * \brief save results to grad
   */
  void Save(SArray<real_t>* grad) const {
    if (pos.empty()) {
      grad->CopyFrom(value.data(), value.size());
    } else {
      for (int i = static_cast<int>(pos.size()); i > 0; --i) {
        if (pos[i-1] != static_cast<unsigned>(-1)) {
          size_t n = pos[i-1]+1;
          if (grad->size() < n) grad->resize(n);
          break;
        }
      }
      for (size_t i = 0; i < pos.size(); ++i) {
        if (pos[i] == static_cast<unsigned>(-1)) continue;
        (*grad)[pos[i]] = value[i];
      }
    }
  }
  SArray<real_t> value;
  SArray<unsigned> pos;
 private:
  bool inited_ = false;
};

/**
 * \brief internal data for the embedding part of FM loss
 */
struct Embedding {
  void Init(int embed_dim, const dmlc::RowBlock<unsigned>& data,
            const SArray<real_t>& model,
            const SArray<int>& model_siz) {
    CHECK(!inited_);
    dim = embed_dim;
    if (dim == 0) return;
    std::vector<unsigned> col_map(model_siz.size());
    unsigned k = 0, p = 0;
    pos.reserve(model_siz.size());
    for (size_t i = 0; i < model_siz.size(); ++i) {
      if (model_siz[i] > 1) {
        CHECK_EQ(model_siz[i], dim + 1);
        pos.push_back(p+1);  // skip the first dim
        col_map[i] = ++k;
      }
      p += model_siz[i];
    }
    CHECK_EQ((size_t)p, model.size());

    value.resize(pos.size() * dim);
    for (size_t i = 0; i < pos.size(); ++i) {
      memcpy(value.data()+i*dim, model.data()+pos[i], dim*sizeof(real_t));
    }

    // pick the columns of data with model_siz = dim + 1
    os_.push_back(0);
    for (size_t i = 0; i < data.size; ++i) {
      for (size_t j = data.offset[i]; j < data.offset[i+1]; ++j) {
        unsigned d = data.index[j];
        unsigned k = col_map[d];
        if (k > 0) {
          idx_.push_back(k-1);
          if (data.value) val_.push_back(data.value[j]);
        }
      }
      os_.push_back(idx_.size());
    }
    X.size = data.size;
    X.offset = BeginPtr(os_);
    X.value = BeginPtr(val_);
    X.index = BeginPtr(idx_);

    XX = X;
    if (X.value) {
      val2_.resize(X.offset[X.size]);
      for (size_t i = 0; i < val2_.size(); ++i) {
        val2_[i] = X.value[i] * X.value[i];
      }
      XX.value = BeginPtr(val2_);
    }
    inited_ = true;
  }

  void Save(SArray<real_t>* grad) const {
    if (dim == 0) return;
    CHECK_EQ(value.size(), pos.size()*dim);
    size_t n = pos.back() + dim;
    if (grad->size() < n) grad->resize(n);
    for (size_t i = 0; i < pos.size(); ++i) {
      CHECK_LE(static_cast<size_t>(pos[i] + dim), grad->size());
      memcpy(grad->data()+pos[i], value.data()+i*dim, dim*sizeof(real_t));
    }
  }

  SArray<real_t> value, XV;
  SArray<unsigned> pos;
  dmlc::RowBlock<unsigned> X, XX;
  int dim;

 private:
  template <typename T>
  const T* BeginPtr(const std::vector<T>& vec) {
    return vec.empty() ? nullptr : vec.data();
  }
  std::vector<real_t> val_, val2_;
  std::vector<size_t> os_;
  std::vector<unsigned> idx_;
  bool inited_ = false;
};

}  // namespace fmloss
}  // namespace difacto
#endif  // _FM_LOSS_UTILS_H_
