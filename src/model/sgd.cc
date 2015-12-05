#include "./sgd.h"
namespace difacto {

KWArgs SGDModel::Init(const KWArgs& kwargs, feaid_t start_id, feaid_t end_id) {
  CHECK_GT(end_id, start_id);
  start_id_ = start_id;
  end_id_ = end_id;
  if (end_id_ - start_id_ < 1e8) {
    dense_ = true;
    model_vec_.resize(end_id - start_id_);
  } else {
    dense_ = false;
  }
  return param_.InitAllowUnknown(kwargs);
}


void SGDModel::Load(dmlc::Stream* fi, bool* has_aux) {
  CHECK_NOTNULL(has_aux);
  CHECK_NOTNULL(fi);
  feaid_t id;
  std::vector<char> tmp((param_.V_dim*2+10)*sizeof(real_t));
  bool has_aux_cur, first = true;
  while (fi->Read(&id, sizeof(id))) {
    int len; fi->Read(&len);
    if (id < start_id_ || id >= end_id_) {
      // skip
      len = len > 0 : len : -len;
      CHECK_LT(len, (int)tmp.size());
      fi->Read(tmp.data(), len);
      continue;
    }
    // load
    id -= start_id_;
    if (dense_) {
      Load(fi, len, &model_vec_[id]);
    } else {
      Load(fi, len, &model_map_[id]);
    }
    // update has_aux
    has_aux_cur = len > 0;
    if (!first) CHECK_EQ(has_aux_cur, *has_aux);
    first = false;
    *has_aux = has_aux_cur;
  }
}

void SGDModel::Save(bool save_aux, dmlc::Stream *fo) const {
  if (dense_) {
    for (feaid_t id = 0; id < (feaid_t)model_vec_.size(); ++i) {
      Save(save_aux, id + start_id_, model_vec_[id], fo);
    }
  } else {
    for (const it& : model_) {
      Save(save_aux, it.first + start_id_, it.second, fo);
    }
  }
}

void SGDModel::Load(dmlc::Stream* fi, int len, SGDEntry* entry) {
  bool has_aux = len > 0;
  len = (len > 0 ? len : - len) / sizeof(real_t);

  CHECK_GE(len, 2);
  fi->Read(&entry->fea_cnt);
  fi->Read(&entry->w);
  len -= 2;

  if (has_aux) {
    CHECK_GE(len, 2);
    fi->Read(&entry->sqrt_g);
    fi->Read(&entry->z);
    len -= 2;
  }

  if (len > 0) {
    CHECK_EQ(len, param_.V_dim * (1 + has_aux));
    entry->V = new real_t[len];
    fi->Read(entry->V, len);
  }
}

void SGDModel::Save(
    bool save_aux, feaid_t id, const SGDEntry& entry, dmlc::Stream *fo) {
  if (!save_aux && entry.V == nullptr && entry.w == 0) {
    // skip empty entry
    return;
  }
  int V_dim = param_.V_dim;
  int V_len = (1 + save_aux) * (entry->V ? V_dim : 0) * sizeof(real_t);
  int len = (1 + save_aux) * 2 * sizeof(real_t) + V_len;
  fo->Write(id);
  fo->Write(len);
  fo->Write(entry.fea_cnt);
  fo->Write(entry.w);
  if (save_aux) {
    fo->Write(entry.sqrt_g);
    fo->Write(entry.z);
  }
  if (V_len) fo->Write(entry.V, V_len);
}


void SGDOptimizer::Get(
    const std::vector<feaid_t>& fea_ids,
    std::vector<T>* weights,
    std::vector<int>* weight_lens) {
  int V_dim = param_.V_dim;
  size_t size = fea_ids.size();
  weights->resize(size * (1 + V_dim));
  weight_lens->resize(V_dim == 0 ? 0 : size);
  int p = 0;
  for (size_t i = 0; i < size; ++i) {
    auto& e = model_[fea_ids[i]];
    weights->at(p++) = e.w;
    if (e.V) {
      memcpy(weights->data()+p, e.V, V_dim*sizeof(real_t));
      p += V_dim;
    }
    if (V_dim != 0) {
      weight_lens->at(i) = (e.V ? V_dim : 0) + 1;
    }
  }
}

void SGDOptimizer::AddCount(
    const std::vector<feaid_t>& fea_ids,
    const std::vector<uint32_t>& fea_cnts) {
  CHECK_EQ(fea_ids.size(), fea_cnts.size());
  for (size_t i = 0; i < fea_ids.size(); ++i) {
    auto& e = model_[fea_ids[i]];
    e.fea_cnt += fea_cnts[i];
    // if (e.V_len == 0 && e.w[0] != 0 && e.fea_cnt > V_threshold_) {
    //     e.InitV(V_dim_, V_init_scale_);
    //   }
    // }
  }
}


void SGDOptimizer::Update(const std::vector<ps::Key>& fea_ids,
                          const std::vector<T>& grad,
                          const std::vector<int>& grad_lens) override {
  size_t size = fea_ids.size();
  bool no_len = gradient_lens.empty();
  if (no_len) { CHECK_EQ(gradients.size(), size); }

  int p = 0;

  for (size_t i = 0; i < size; ++i) {
    auto& e = model_[fea_ids[i]];
    UpdateW(gradients[p], &e);
    if (!no_len && gradient_lens[i] > 1) {
      int n = gradient_lens[i] - 1;
      UpdateV(gradients.data() + p, n, &e);
      p += n;
    }
  }
  CHECK_EQ((size_t)p, gradients.size());
}


// T w_alpha_, w_beta_, V_alpha_, V_beta_;
// T V_init_scale_;
// int V_dim_, V_threshold_;

void SGDOptimizer::UpdateW(real_t gw, SGDEntry* e) {
  T w = e->w[0];
  T cg = e->w[1];
  // update w[1]
  gw += w * w_l2_;
  e->w[1] = sqrt(cg * cg + gw * gw);
  // update w[2]
  e->w[2] -= gw - (e->w[1] - cg) / w_alpha_ * w;
  // update w[0] by soft shrinkage
  T z = e->w[2];
  if (z <= w_l1_ && z >= - w_l1_) {
    e->w[0] = 0;
  } else {
    T eta = (w_beta + e->w[1]) / w_alpha;
    e->w[0] = (z > 0 ? z - w_l1_ : z + w_l1_) / eta;
  }
  // update statistics
  if (w == 0 && e->w[0] != 0) {
    ++ new_w_;
    if (e.V_len == 0 && e.fea_cnt > V_threshold_) {
      e.InitV(V_dim_, V_init_scale_);
    }
  } else if (w != 0 && e->w[0] == 0) {
    -- new_w_;
  }
}

void SGDOptimizer::UpdateV(real_t const* gV, SGDEntry* e) {

  for (int i = 0; i < n; ++i) {
    T g = gV[i] + V_l2_ * e->V[i];
    T cg = e->V[i+len];
    e->V[i+len] = sqrt(cg * cg + g * g);
    float eta = V_alpha_ / ( e->V[i+len] + V_beta_ );
    e->V[i] -= eta * g;
  }
}


void SGDOptimizer::InitV(SGDEntry* e) {
  // /** \brief init V */
  // void InitV(int len, T init_scale) {
  //   CHECK_EQ(V_len, 0) << "already inited";
  //   V_len = len;
  //   V = new T[len * 2];
  //   for (int i = 0; i < len; ++i) {
  //     V[i] = (rand() / static_cast<T>(RAND_MAX) - 0.5) * init_scale;
  //   }
  //   memset(V+len, 0, len*sizeof(T));
  // }
}

}  // namespace difacto
