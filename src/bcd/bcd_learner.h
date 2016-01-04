#ifndef DEFACTO_LEARNER_BCD_LEARNER_H_
#define DEFACTO_LEARNER_BCD_LEARNER_H_
#include "difacto/learner.h"
#include "difacto/node_id.h"
#include "dmlc/data.h"
#include "data/chunk_iter.h"
#include "data/data_store.h"
#include "common/kv_union.h"
#include "common/spmt.h"
#include "./bcd_param.h"
#include "./bcd_job.h"
#include "./bcd_utils.h"
#include "loss/fm_loss_delta.h"
namespace difacto {

class BCDLearner : public Learner {
 public:
  BCDLearner() {
  }
  virtual ~BCDLearner() {
  }

  KWArgs Init(const KWArgs& kwargs) override {
  }
 protected:

  void RunScheduler() override {
    // load and convert data
    bool has_val = param_.data_val.size() != 0;
    CHECK(param_.data_in.size());
    IssueJobToWorkers(bcd::JobArgs::kPrepareTrainData, param_.data_in);

    if (has_val) {
      IssueJobToWorkers(bcd::JobArgs::kPrepareValData, param_.data_val);
    }

    epoch_ = 0;
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {
      IssueJobToWorkers(bcd::JobArgs::kTraining);
      if (has_val) IssueJobToWorkers(bcd::JobArgs::kValidation);
    }
  }

  void Process(const std::string& args, std::string* rets) {
    bcd::JobArgs job(args);

    if (job.type == bcd::JobArgs::kPrepareValData ||
        job.type == bcd::JobArgs::kPrepareTrainData) {
      bcd::PrepDataRets prets;
      PrepareData(job, &prets);
      prets.SerializeToString(rets);
    } else if (job.type == bcd::JobArgs::kTraining ||
               job.type == bcd::JobArgs::kValidation) {
      // IterateFeatureBlocks(job, rets);
    }

    // if (job.type == kSaveModel) {
    // } else if (job.type == kLoadModel) {
    // } else {
    //   ProcessFile(job);
    // }
  }

 private:
  /**
   * \brief sleep for a moment
   * \param ms the milliseconds (1e-3 sec) for sleeping
   */
  inline void Sleep(int ms = 1000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  }

  /**
   * \brief send jobs to workers and wait them finished.
   */
  void IssueJobToWorkers(int job_type,
                         const std::string& filename = "") {
    // job_type_ = job_type;
    // for (auto& cb : before_epoch_callbacks_) cb();
    // int nworker = model_store_->NumWorkers();
    // std::vector<Job> jobs(nworker);
    // for (int i = 0; i < nworker; ++i) {
    //   jobs[i].type = job_type;
    //   jobs[i].filename = filename;
    //   jobs[i].num_parts = nworker;
    //   jobs[i].part_idx = i;
    //   jobs[i].node_id = NodeID::Encode(NodeID::kWorkerGroup, i);
    // }
    // job_tracker_->Add(jobs);
    // while (job_tracker_->NumRemains() != 0) { Sleep(); }
    // for (auto& cb : epoch_callbacks_) cb();
  }

  void PrepareData(const bcd::JobArgs& job, bcd::PrepDataRets* rets) {
    // read and process a 512MB chunk each time
    ChunkIter reader(job.filename, param_.data_format,
                     job.part_idx, job.num_parts, 1<<28);

    // std::vector<feaid_t> *all_ids = nullptr, *ids;
    // std::vector<real_t> *all_cnt = nullptr, *cnt;

    int nbit = param_.num_feature_group_bits;
    CHECK_EQ(nbit % 4, 0) << "should be 0, 4, 8, ...";
    CHECK_LE(nbit, 16);

    std::vector<real_t> feablk_occur(1<<nbit);
    real_t row_counted = 0;
    SArray<feaid_t> all_feaids;
    SArray<real_t> all_feacnts;
    while (reader.Next()) {
      // count statistic infos
      auto rowblk = reader.Value();
      int skip = 10;  // only count 10% data
      for (size_t i = 0; i < rowblk.size; i+=skip) {
        for (size_t j = rowblk.offset[i]; j < rowblk.offset[i+1]; ++j) {
          feaid_t f = rowblk.index[j];
          ++feablk_occur[f-((f>>nbit)<<nbit)];
        }
        ++row_counted;
      }

      // map feature id into continous intergers and transpose to easy slice a
      // column block
      std::shared_ptr<std::vector<feaid_t>> feaids(new std::vector<feaid_t>());
      std::shared_ptr<std::vector<real_t>> feacnt(new std::vector<real_t>());


      Localizer lc(-1, 2);
      auto compacted = new dmlc::data::RowBlockContainer<unsigned>();
      auto transposed = new dmlc::data::RowBlockContainer<unsigned>();

      lc.Compact(rowblk, compacted, feaids.get(), feacnt.get());
      SpMT::Transpose(compacted->GetBlock(), transposed, feaids->size(), 2);
      delete compacted;

      // push into data store
      SArray<feaid_t> sfeaids(feaids);
      auto id = std::to_string(num_data_blks_++) + "_";
      SharedRowBlockContainer<unsigned> data(&transposed);
      data_store_->Push(id + "data", data);
      data_store_->Push(id + "feaids", sfeaids);
      data_store_->Push(id + "label", rowblk.label, rowblk.size);
      delete transposed;

      // merge ids and counts
      SArray<real_t> sfeacnts(feacnt);
      if (all_feaids.empty()) {
        all_feaids = sfeaids;
        all_feacnts = sfeacnts;
      } else {
        SArray<feaid_t> new_feaids;
        SArray<real_t> new_feacnts;
        // TODO
        // KVUNion(sfeaids, sfeacnts, all_feaids, all_feacnts,
        // &new_feaids, &new_feacnts, 1, PLUS, 2);
        all_feaids = new_feaids;
        all_feacnts = new_feacnts;
      }

      // init predict
      data_store_->Push(id + "predict", SArray<real_t>(rowblk.size, 0));
    }

    // push the feature ids and feature counts to the servers
    data_store_->Push("feaids", all_feaids);
    int t = model_store_->Push(
        Store::kFeaCount, all_feaids, all_feacnts, SArray<int>());
    model_store_->Wait(t);

    // report statistics to the scheduler
    if (row_counted) {
      for (real_t& o : feablk_occur) o /= row_counted;
    }
    CHECK_NOTNULL(rets)->feablk_avg = feablk_occur;


  }

  void BuildFeatureMap(const bcd::JobArgs& job) {
    // pull the aggregated feature counts from the servers
    SArray<feaid_t> feaids;
    data_store_->Pull("feaids", &feaids);
    SArray<real_t> feacnt;
    int t = model_store_->Pull(Store::kFeaCount, feaids, &feacnt, nullptr);
    model_store_->Wait(t);

    // remove the filtered features
    SArray<feaid_t> filtered;
    CHECK_EQ(feacnt.size(), feaids.size());
    for (size_t i = 0; i < feaids.size(); ++i) {
      if (feacnt[i] > param_.tail_feature_filter) {
        filtered.push_back(feaids[i]);
      }
    }
    data_store_->Push("feaids", filtered);

    // partition feature space and save the feature block locations
    int nbit = param_.num_feature_group_bits;
    feablk_pos_.resize(num_data_blks_ + 1);
    CHECK(job.fea_blk_ranges.size());
    bcd::FindPosition(filtered, job.fea_blk_ranges, &feablk_pos_[0]);

    // build the map for each data block
    SArray<int> map_idx(filtered.size());
    for (size_t i = 0; i < map_idx.size(); ++i) {
      map_idx[i] = i + 1;
    }
    for (int b = 0; b < num_data_blks_; ++b) {
      auto id = std::to_string(b) + "_";
      data_store_->NextPullHint(id + "feaids");
    }
    for (int b = 0; b < num_data_blks_; ++b) {
      auto id = std::to_string(b) + "_";
      SArray<feaid_t> blk_feaids;
      data_store_->Pull(id + "feaids", &blk_feaids);
      data_store_->Remove(id + "feaids");
      bcd::FindPosition(blk_feaids, job.fea_blk_ranges, &feablk_pos_[b+1]);

      SArray<int> blk_feamap;
      // TODO
      // KVMatch(filtered, map_idx, blk_feaids, &blk_feamap, 1, ASSIGN, 2);
      for (int& m : blk_feamap) --m;
      data_store_->Push(id + "feamap", blk_feamap);
    }
  }

  void IterateFeatureBlocks(const bcd::JobArgs& job, bcd::IterFeaBlkRets* rets) {
    CHECK(job.fea_blks.size());
    // hint for data prefetch
    for (int f : job.fea_blks) {
      data_store_->NextPullHint("feaids", feablk_pos_[0][f]);
      data_store_->NextPullHint(std::to_string(f) + "_len");
      for (int d = 0; d < num_data_blks_; ++d) {
        auto id = std::to_string(d) + "_";
        data_store_->NextPullHint(id + "data", feablk_pos_[d+1][f]);
        data_store_->NextPullHint(id + "feamap", feablk_pos_[d+1][f]);
        data_store_->NextPullHint(id + "predict");
      }
    }

    size_t nfeablk = job.fea_blks.size();
    int tau = 0;
    FeatureBlockTracker feablk_tracker(nfeablk);
    for (size_t i = 0; i < nfeablk; ++i) {
      // the logic is as following
      //
      // 1. calculate gradident
      // 2. push gradients to servers, so servers will update the weight
      // 3. once the push is done, pull the changes for the weights back from
      //    the servers
      // 4. once the pull is done update the prediction
      //
      // however, two things make the implementation is not so intuitive.
      //
      // 1. we need to iterate the data block one by one for both calcluating
      // gradient and update prediction
      // 2. we used callbacks to avoid to be blocked by the push and pull.
      int f = job.fea_blks[i];

      // 1. calculate gradient
      SArray<int> grad_len;
      data_store_->Pull(std::to_string(f) + "_len", &grad_len);
      size_t n = 0; for (int l : grad_len) n += l;
      SArray<real_t> grad(n);
      CalcGrad(f, grad_len, &grad);

      // fetech id for communication
      SArray<feaid_t> feaids;
      data_store_->Pull("feaids", &feaids, feablk_pos_[0][f]);

      // 3. once push is done, pull the changes for the weights
      // this callback will be called when the push is finished
      auto push_callback = [this, f, feaids, i, &feablk_tracker]() {
        // must use pointer here
        SArray<real_t>* w_delta = new SArray<real_t>();
        SArray<int>* w_delta_len = new SArray<int>();
        // 4. once the pull is done, update the prediction
        // the callback will be called when the pull is finished
        auto pull_callback = [this, f, w_delta, w_delta_len, i, &feablk_tracker]() {
          data_store_->Push(std::to_string(f) + "_len", *w_delta_len);
          UpdtPred(f, *w_delta, *w_delta_len);
          delete w_delta;
          delete w_delta_len;
          feablk_tracker.Finish(i);
        };
        // pull the changes of w from the servers
        int t = model_store_->Pull(Store::kWeight, feaids, w_delta, w_delta_len, pull_callback);
      };
      // 2. push gradient to the servers
      model_store_->Push(Store::kGradient, feaids, grad, grad_len, push_callback);


      if (i >= tau) feablk_tracker.Wait(i - tau);
    }
    for (int i = nfeablk - tau; i < nfeablk ; ++i) feablk_tracker.Wait(i);
  }

  void CalcGrad(int fea_blk_id, const SArray<int>& grad_len, SArray<real_t>* grad) {
    for (int d = 0; d < num_data_blks_; ++d) {
      // load data
      Range fea_blk_pos = feablk_pos_[d+1][fea_blk_id];
      auto id = std::to_string(d) + "_";
      SArray<int> fea_map;
      data_store_->Pull(id + "feamap", &fea_map, fea_blk_pos);

      SArray<int> blk_grad_len(fea_map.size());
      for (size_t i = 0; i < fea_map.size(); ++i) {
        blk_grad_len[i] = grad_len[fea_map[i]];
      }

      SharedRowBlockContainer<unsigned> data;
      data_store_->Pull(id + "data", &data, fea_blk_pos);

      FMLossDelta* loss = new FMLossDelta();
      // loss->Init();
      SArray<real_t> pred;
      data_store_->Pull(id + "predict", &pred);

      // calc grad
      SArray<real_t> blk_grad;
      loss->CalcGrad(data.GetBlock(),
                     {SArray<char>(pred), SArray<char>(blk_grad_len)},
                     &blk_grad);

      real_t const* cur_blk_grad = blk_grad.data();
      real_t* cur_grad = grad->data();
      for (size_t i = 0; i < blk_grad_len.size(); ++i) {
        if (i > 0 && fea_map[i] > fea_map[i-1]+1) {
          for (int j = fea_map[i-1]+1; j < fea_map[i]; ++j) cur_grad += grad_len[j];
        }
        memcpy(cur_grad, cur_blk_grad, blk_grad_len[i]*sizeof(real_t));
        cur_grad += blk_grad_len[i];
        cur_blk_grad += blk_grad_len[i];
      }
      CHECK(cur_grad == grad->end());
      CHECK(cur_blk_grad == blk_grad.end());
    }
  }

  void UpdtPred(int fea_blk_id, const SArray<real_t>& w_delta,
                const SArray<int>& w_delta_len) {
    for (int d = 0; d < num_data_blks_; ++d) {
      // load data
      Range fea_blk_pos = feablk_pos_[d+1][fea_blk_id];
      auto id = std::to_string(d) + "_";
      SArray<int> fea_map;
      data_store_->Pull(id + "feamap", &fea_map, fea_blk_pos);

      size_t n = 0;
      SArray<int> blk_w_len(fea_map.size());
      for (size_t i = 0; i < fea_map.size(); ++i) {
        blk_w_len[i] = w_delta_len[fea_map[i]];
        n += blk_w_len[i];
      }
      SArray<real_t> blk_w(n);
      real_t* cur_blk_w = blk_w.data();
      real_t const* cur_w = w_delta.data();
      for (size_t i = 0; i < blk_w_len.size(); ++i) {
        if (i > 0 && fea_map[i] > fea_map[i-1]+1) {
          for (int j = fea_map[i-1]+1; j < fea_map[i]; ++j) cur_w += w_delta_len[j];
        }
        memcpy(cur_blk_w, cur_w, blk_w_len[i]*sizeof(real_t));
        cur_blk_w += blk_w_len[i];
        cur_w += blk_w_len[i];
      }
      CHECK(cur_w == w_delta.end());
      CHECK(cur_blk_w == blk_w.end());

      SharedRowBlockContainer<unsigned> data;
      data_store_->Pull(id + "data", &data, fea_blk_pos);

      FMLossDelta* loss = new FMLossDelta();
      // loss->Init();
      SArray<real_t> old_pred, pred;
      data_store_->Pull(id + "predict", &old_pred);

      loss->Predict(data.GetBlock(), {SArray<char>(old_pred), SArray<char>(blk_w),
              SArray<char>(blk_w_len)}, &pred);
      data_store_->Push(id + "predict", pred);
    }
  }

  /** \brief the current epoch */
  int epoch_;
  /** \brief the current job type */
  int job_type_;
  int num_data_blks_ = 0;
  /** \brief the model store*/
  Store* model_store_ = nullptr;
  /** \brief data store */
  DataStore* data_store_ = nullptr;
  /** \brief parameters */
  BCDLearnerParam param_;

  std::vector<std::vector<Range>> feablk_pos_;

  /**
   * \brief monitor if or not a feature block is finished
   */
  class FeatureBlockTracker {
   public:
    FeatureBlockTracker(int num_blks) : done_(num_blks) { }
    /** \brief mark id as finished */
    void Finish(int id) {
      mu_.lock();
      done_[id] = 1;
      mu_.unlock();
      cond_.notify_all();
    }
    /** \brief block untill id is finished */
    void Wait(int id) {
      std::unique_lock<std::mutex> lk(mu_);
      cond_.wait(lk, [this, id] {return done_[id] == 1; });
    }
   private:
    std::mutex mu_;
    std::condition_variable cond_;
    std::vector<int> done_;
  };

};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_BCD_LEARNER_H_
