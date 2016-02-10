/**
 * Copyright (c) 2015 by Contributors
 */
#include <string.h>
#include "./sgd_updater.h"
#include "difacto/store.h"
namespace difacto {

void SGDModel::Init(int V_dim, feaid_t start_id, feaid_t end_id) {
  V_dim_ = V_dim;
  CHECK_GT(end_id, start_id);
  start_id_ = start_id;
  end_id_ = end_id;
  if (end_id_ - start_id_ < 1e8) {
    dense_ = true;
    model_vec_.resize(end_id - start_id_);
  } else {
    dense_ = false;
  }
}


void SGDModel::Load(dmlc::Stream* fi, bool* has_aux) {
  CHECK_NOTNULL(has_aux);
  CHECK_NOTNULL(fi);
  feaid_t id;
  std::vector<char> tmp((V_dim_*2+10)*sizeof(real_t));
  bool has_aux_cur, first = true;
  while (fi->Read(&id, sizeof(id))) {
    int len; fi->Read(&len);
    if (id < start_id_ || id >= end_id_) {
      // skip
      len = len > 0 ? len : -len;
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
    for (feaid_t id = 0; id < (feaid_t)model_vec_.size(); ++id) {
      Save(save_aux, id + start_id_, model_vec_[id], fo);
    }
  } else {
    for (const auto& it : model_map_) {
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
    CHECK_EQ(len, V_dim_ * (1 + has_aux));
    entry->V = new real_t[len];
    fi->Read(entry->V, len);
  }
}

void SGDModel::Save(bool save_aux, feaid_t id,
                    const SGDEntry& entry, dmlc::Stream *fo) const {
  if (!save_aux && entry.V == nullptr && entry.w == 0) {
    // skip empty entry
    return;
  }
  int V_len = (1 + save_aux) * (entry.V ? V_dim_ : 0) * sizeof(real_t);
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

KWArgs SGDUpdater::Init(const KWArgs& kwargs) {
  auto remain = param_.InitAllowUnknown(kwargs);
  model_.Init(param_.V_dim, 0, std::numeric_limits<feaid_t>::max());
  remain.push_back(std::make_pair("V_dim", std::to_string(param_.V_dim)));
  return remain;
}



void SGDUpdater::Get(const SArray<feaid_t>& fea_ids,
                     int val_type,
                     SArray<real_t>* weights,
                     SArray<int>* offsets) {
  CHECK_EQ(val_type, Store::kWeight);
  int V_dim = param_.V_dim;
  size_t size = fea_ids.size();
  weights->resize(size * (1 + V_dim));
  offsets->resize(V_dim == 0 ? 0 : size+1);
  (*offsets)[0] = 0;
  int p = 0;
  for (size_t i = 0; i < size; ++i) {
    auto& e = model_[fea_ids[i]];
    (*weights)[p++] = e.w;
    if (e.V) {
      memcpy(weights->data()+p, e.V, V_dim*sizeof(real_t));
      p += V_dim;
    }
    if (V_dim != 0) (*offsets)[i+1] = p;
  }
  weights->resize(p);
}

void SGDUpdater::Update(const SArray<feaid_t>& fea_ids,
                        int value_type,
                        const SArray<real_t>& values,
                        const SArray<int>& offsets) {
  if (value_type == Store::kFeaCount) {
    CHECK_EQ(fea_ids.size(), values.size());
    for (size_t i = 0; i < fea_ids.size(); ++i) {
      auto& e = model_[fea_ids[i]];
      e.fea_cnt += values[i];
      if (e.V == nullptr && e.w != 0 && e.fea_cnt > param_.V_threshold) {
        InitV(&e);
      }
    }
  } else if (value_type == Store::kGradient) {
    CHECK(has_aux_) << "no aux data";
    size_t size = fea_ids.size();
    bool w_only = offsets.empty();
    if (w_only) {
      CHECK_EQ(values.size(), size);
    } else {
      CHECK_EQ(offsets.size(), size+1);
      CHECK_EQ(offsets.back(), static_cast<int>(values.size()));
    }
    for (size_t i = 0; i < size; ++i) {
      auto& e = model_[fea_ids[i]];
      UpdateW(values[offsets[i]], &e);
      if (!w_only && offsets[i+1] > offsets[i]+1) {
        CHECK_EQ(offsets[i+1], offsets[i]+1);
        UpdateV(values.data() + offsets[i] + 1, &e);
      }
    }
  } else {
    LOG(FATAL) << ".....";
  }
  // TODO(mli)
  // Progress prog;
  // prog.new_w() = new_w_;
  // prog.new_V() = new_V_;
  // ppmonitor_->Add(prog);
}


void SGDUpdater::UpdateW(real_t gw, SGDEntry* e) {
  real_t sg = e->sqrt_g;
  real_t w = e->w;
  // update sqrt_g
  gw += w * param_.l2;
  e->sqrt_g = sqrt(sg * sg + gw * gw);
  // update z
  e->z -= gw - (e->sqrt_g - sg) / param_.lr * w;
  // update w by soft shrinkage
  real_t z = e->z;
  real_t l1 = param_.l1;
  if (z <= l1 && z >= - l1) {
    e->w = 0;
  } else {
    real_t eta = (param_.lr_beta + e->sqrt_g) / param_.lr;
    e->w = (z > 0 ? z - l1 : z + l1) / eta;
  }
  // update statistics
  if (w == 0 && e->w != 0) {
    ++new_w_;
    if (e->V == nullptr && e->fea_cnt > param_.V_threshold) {
      InitV(e);
    }
  } else if (w != 0 && e->w == 0) {
    --new_w_;
  }
}

void SGDUpdater::UpdateV(real_t const* gV, SGDEntry* e) {
  int n = param_.V_dim;
  for (int i = 0; i < n; ++i) {
    real_t g = gV[i] + param_.V_l2 * e->V[i];
    real_t cg = e->V[i+n];
    e->V[i+n] = sqrt(cg * cg + g * g);
    float eta = param_.V_lr / (e->V[i+n] + param_.V_lr_beta);
    e->V[i] -= eta * g;
  }
}

void SGDUpdater::InitV(SGDEntry* e) {
  int n = param_.V_dim;
  e->V = new real_t[n*2];
  for (int i = 0; i < n; ++i) {
    e->V[i] = (rand_r(&param_.seed) / (real_t)RAND_MAX - 0.5) * param_.V_init_scale;
  }
  memset(e->V+n, 0, n*sizeof(real_t));
  new_V_ += n;
}

}  // namespace difacto
