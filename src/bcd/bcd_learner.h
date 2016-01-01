#ifndef DEFACTO_LEARNER_BCD_LEARNER_H_
#define DEFACTO_LEARNER_BCD_LEARNER_H_
#include "difacto/learner.h"
#include "difacto/node_id.h"
#include "dmlc/parameter.h"
#include "dmlc/data.h"
#include "data/chunk_iter.h"
#include "data/data_store.h"
#include "common/kv_union.h"
#include "common/spmt.h"
namespace difacto {

struct BCDLearnerParam : public dmlc::Parameter<BCDLearnerParam> {
  /**
   * \brief type of task,
   * - train: the training task, which is the default
   * - predict: the prediction task
   */
  std::string task;
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /**
   * \brief The optional validation dataset for a training task, either a
   *  filename or a directory
   */
  std::string data_val;
  /** \brief the data format. default is libsvm */
  std::string data_format;
  /** \brief the directory for the data chache */
  std::string data_cache;
  /** \brief the model output for a training task */
  std::string model_out;
  /**
   * \brief the model input
   * should be specified if it is a prediction task, or a training
   */
  std::string model_in;
  /**
   * \brief the filename for prediction output.
   *  should be specified for a prediction task.
   */
  std::string pred_out;
  /** \brief type of loss, defaut is fm*/
  std::string loss;
  /** \brief the maximal number of data passes, defaut is 20 */
  int max_num_epochs;

  /** \brief controls the number of feature blocks, default is 4 */
  float block_ratio;

  /** \brief if or not process feature blocks in a random order, default is true */
  int random_block;

  /** \brief the number of bit used to encode the feature group, default is 0 */
  int feature_group_nbit;
  float neg_sampling;

  DMLC_DECLARE_PARAMETER(BCDLearnerParam) {
    DMLC_DECLARE_FIELD(task).set_default("train");
    DMLC_DECLARE_FIELD(data_format).set_default("libsvm");
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_val).set_default("");
    DMLC_DECLARE_FIELD(model_out).set_default("");
    DMLC_DECLARE_FIELD(model_in).set_default("");
    DMLC_DECLARE_FIELD(pred_out).set_default("");
    DMLC_DECLARE_FIELD(loss).set_default("fm");
    DMLC_DECLARE_FIELD(max_num_epochs).set_default(20);
    DMLC_DECLARE_FIELD(random_block).set_default(1);
    DMLC_DECLARE_FIELD(feature_group_nbit).set_default(0);
    DMLC_DECLARE_FIELD(block_ratio).set_default(4);
    DMLC_DECLARE_FIELD(data_cache).set_default("/tmp/cache_difacto_");
  }
};

struct BCDJob {
  /** \brief construct from a string */
  BCDJob(const std::string& str) { ParseFromString(str); }

  void SerializeToString(std::string* string) {
    // TODO
  }

  void ParseFromString(const std::string& str) {
    // TODO
  }
  static const int kLoadModel = 1;
  static const int kSaveModel = 2;
  static const int kTraining = 3;
  static const int kValidation = 4;
  static const int kPrediction = 5;
  static const int kPrepareTrainData = 6;
  static const int kPrepareValData = 7;
  /** \brief job type */
  int type;
  /** \brief filename  */
  std::string filename;
  /** \brief number of partitions of this file */
  int num_parts;
  /** \brief the part will be processed */
  int part_idx;
  /** \brief the order to process feature blocks */
  std::vector<int> fea_blks;
};

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
    IssueJobToWorkers(BCDJob::kPrepareTrainData, param_.data_in);

    if (has_val) {
      IssueJobToWorkers(BCDJob::kPrepareValData, param_.data_val);
    }

    epoch_ = 0;
    for (; epoch_ < param_.max_num_epochs; ++epoch_) {
      IssueJobToWorkers(BCDJob::kTraining);
      if (has_val) IssueJobToWorkers(BCDJob::kValidation);
    }
  }

  void Process(const std::string& args, std::string* rets) {
    BCDJob job(args);

    if (job.type == BCDJob::kPrepareValData ||
        job.type == BCDJob::kPrepareTrainData) {
      // PrepareData(job, rets);
    } else if (job.type == BCDJob::kTraining ||
               job.type == BCDJob::kValidation) {
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

  // void PrepareData(const Job& job) {
  //   // read a 512MB chunk each time
  //   ChunkIter reader(
  //       job.filename, param_.data_format, job.part_idx, job.num_parts,
  //       1<<28);

  //   std::vector<feaid_t> *all_ids = nullptr, *ids;
  //   std::vector<real_t> *all_cnt = nullptr, *cnt;
  //   while (reader.Next()) {
  //     // map feature id into continous intergers and transpose to easy slice a
  //     // column block
  //     ids = new std::vector<feaid_t>();
  //     cnt = new std::vector<real_t>();
  //     Localizer lc(-1, 2);
  //     dmlc::data::RowBlockContainer<unsigned> compacted, transposed;
  //     lc.Compact(reader.Value(), &compacted, ids, cnt);
  //     SpMT::Transpose(compacted.GetBlock(), &transposed, ids->size(), 2);

  //     int id = (num_data_blks_++) * kDataBOS_;
  //     data_store_->Push(id, transposed.GetBlock());
  //     data_store_->Push(id + 1, ids->data(), ids->size());
  //     data_store_->Push(id + 2, compacted.label.data(), compacted.label.size());

  //     // merge ids and counts
  //     if (all_ids == nullptr) {
  //       all_ids = ids;
  //       all_cnt = cnt;
  //     } else {
  //       auto old_ids = all_ids;
  //       auto old_cnt = all_cnt;
  //       all_ids = new std::vector<feaid_t>();
  //       all_cnt = new std::vector<real_t>();
  //       KVUnion(*old_ids, *old_cnt, *ids, *cnt, all_ids, all_cnt, 1, PLUS, 2);
  //       delete old_ids;
  //       delete old_cnt;
  //       delete ids;
  //       delete cnt;
  //     }
  //   }
  // }


  // void IterateFeatureBlocks(const Job& job) {

  // }

  // void ProcessFile(const Job& job) {

  // }

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
};

}  // namespace difacto
#endif  // DEFACTO_LEARNER_BCD_LEARNER_H_
