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
    // read a 512MB chunk each time
    ChunkIter reader(
        job.filename, param_.data_format, job.part_idx, job.num_parts,
        1<<28);

    std::vector<feaid_t> *all_ids = nullptr, *ids;
    std::vector<real_t> *all_cnt = nullptr, *cnt;

    int nbit = param_.num_feature_group_bits;
    CHECK_LT(nbit, 18);
    std::vector<real_t> feablk_occur(1<<nbit);
    real_t row_counted = 0;
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
      ids = new std::vector<feaid_t>();
      cnt = new std::vector<real_t>();
      Localizer lc(-1, 2);
      dmlc::data::RowBlockContainer<unsigned> compacted, transposed;
      lc.Compact(rowblk, &compacted, ids, cnt);
      SpMT::Transpose(compacted.GetBlock(), &transposed, ids->size(), 2);

      int id = (num_data_blks_++) * kDataBOS_;
      data_store_->Push(id, transposed.GetBlock());
      data_store_->Push(id + 1, ids->data(), ids->size());
      data_store_->Push(id + 2, compacted.label.data(), compacted.label.size());

      // merge ids and counts
      if (all_ids == nullptr) {
        all_ids = ids;
        all_cnt = cnt;
      } else {
        auto old_ids = all_ids;
        auto old_cnt = all_cnt;
        all_ids = new std::vector<feaid_t>();
        all_cnt = new std::vector<real_t>();
        KVUnion(*old_ids, *old_cnt, *ids, *cnt, all_ids, all_cnt, 1, PLUS, 2);
        delete old_ids;
        delete old_cnt;
        delete ids;
        delete cnt;
      }
    }

    if (row_counted) {
      for (real_t& o : feablk_occur) o /= row_counted;
    }
    CHECK_NOTNULL(rets)->feablk_avg = feablk_occur;
  }

  void BuildFeatureMap(const bcd::JobArgs& job) {
    // pull the feature count from the servers
    std::vector<real_t> cnts;
    std::vector<int> lens;
    int t = model_store_->Pull(Store::kFeaCount, fea_ids_, &cnts, &lens);
    model_store_->Wait(t);

    // remove the filtered features
    std::vector<feaid_t>* filtered = new std::vector<feaid_t>();
    CHECK_EQ(cnts.size(), fea_ids_->size());
    for (size_t i = 0; i < fea_ids_->size(); ++i) {
      if (cnts[i] > param_.tail_feature_filter) {
        filtered->push_back(fea_ids_->at(i));
      }
    }
    // fea_ids_.reset(filtered);
    std::vector<int> map_idx(filtered->size());
    for (size_t i = 0; i < map_idx.size(); ++i) {
      map_idx[i] = i + 1;
    }

    // build the map for each data block
    for (int b = 0; b < num_data_blks_; ++b) {
      int id = b * kDataBOS_ + 1;
      data_store_->NextPullHint(id);
    }
    for (int b = 0; b < num_data_blks_; ++b) {
      int id = b * kDataBOS_ + 1;
      feaid_t* tmp;
      size_t n = data_store_->Pull(id, &tmp);
      std::vector<int> blk_map(n);
      std::vector<feaid_t> blk_fea(n);
      memcpy(blk_fea.data(), tmp, n * sizeof(feaid_t));
      data_store_->Remove(id);

      KVMatch(*filtered, map_idx, blk_fea, &blk_map, 1, ASSIGN, 2);
      for (int& m : blk_map) --m;
      // data_store_->Push(id, blk_map.data());
    }

    // store the feature block positions

  }

  void IterateFeatureBlocks(const bcd::JobArgs& job, bcd::IterFeaBlkRets* rets) {


  }


  static const int kDataBOS_ = 5;
  /** \brief the current epoch */
  int epoch_;
  /** \brief the current job type */
  int job_type_;
  int num_data_blks_ = 0;
  /** \brief the model store*/
  Store* model_store_ = nullptr;
  /** \brief data store */
  DataStore* data_store_ = nullptr;
  /** \brief the loss*/
  Loss* loss_ = nullptr;
  /** \brief parameters */
  BCDLearnerParam param_;

  std::vector<Range> feablk_range_;
  std::shared_ptr<std::vector<feaid_t>> fea_ids_;
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_BCD_LEARNER_H_
