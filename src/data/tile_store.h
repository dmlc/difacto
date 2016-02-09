#ifndef _TILE_STORE_H_
#define _TILE_STORE_H_
#include "dmlc/data.h"
#include "difacto/sarray.h"
#include "./shared_row_block_container.h"
#include "./data_store.h"
namespace difacto {
/**
 * \brief a sliced block of a large matrix
 *
 * assume the we evenly partition the following 4x4 matrix into 2 2x4 tiles
 * \code
 * 1..4
 * ..2.
 * .3..
 * 2..1
 * \endcode
 *
 * then the second tile  is
 *
 * \code
 * .3.
 * 2.1
 * \endcode
 */
struct Tile {
  /** \brief the map to the column id on the original matrix */
  SArray<int> colmap;
  /** \brief the transposed data to make slice efficient */
  SharedRowBlockContainer<unsigned> data;
};

class TileBuilder;

class TileStore {
 public:
  TileStore(DataStore* data) {
    data_ = CHECK_NOTNULL(data);
  }
  friend class TileBuilder;

  void Prefetch(int rowblk_id, int colblk_id) {
    auto id = std::to_string(rowblk_id);
    auto rg = colblk_pos_[rowblk_id][colblk_id];
    // data_->Prefetch(id + "_data", rg);
    // data_->Prefetch(id + "_colmap", rg);
  }

  void Fetch(int rowblk_id, int colblk_id, Tile* tile) {
    auto id = std::to_string(rowblk_id);
    auto rg = colblk_pos_[rowblk_id][colblk_id];
    CHECK_NOTNULL(tile);
    // data_->Fetch(id + "_data", &tile->data, rg);
    // data_->Fetch(id + "_colmap", &tile->colmap, rg);
  }

 private:
  DataStore* data_;
  std::vector<std::vector<Range>> colblk_pos_;
};

}  // namespace difacto
#endif  // _TILE_STORE_H_



//   std::vector<std::string> GetRowBlockKeys(const std::string& key) {
//     return {key+"_offset", key+"_label", key+"_weight", key+"_index", key+"_value"};
//   }

//   inline bool IsRowBlockKey(const std::string& key) {
//     return rowblk_keys_.count(key) != 0;
//   }
//   std::unordered_set<std::string> rowblk_keys_;


//     if (IsRowBlockKey(key)) {
//       auto keys = GetRowBlockKeys(key);
//       Range rg1 = range == Range::All() ? range
//                   : Range(range.begin, range.end+1) * sizeof(size_t);
//       store_->Prefetch(keys[0], rg1, [this, key, keys, range](const SArray<char>& data) {
//           SArray<size_t> offset(data);
//           Range rg = Range(offset.front(), offset.back());
//           store_->Prefetch(keys[3], GetCharRange(keys[3], rg));
//           store_->Prefetch(keys[4], GetCharRange(keys[4], rg));
//         });
//       store_->Prefetch(keys[1], GetCharRange(keys[1], range));
//       store_->Prefetch(keys[2], GetCharRange(keys[2], range));
//     } else {
//     }
//   }

//     if (IsRowBlockKey(key)) {
//       for (const auto& s : GetRowBlockKeys(key)) {
//         store_->Remove(s);
//       }
//     } else {


// #include "./shared_row_block_container.h"
//   /**
//    * \brief copy a rowblock into the store
//    *
//    * @param key the unique key
//    * @param data the rowblock
//    */
//   template <typename T>
//   void Store(const std::string& key, const dmlc::RowBlock<T>& data) {
//     SharedRowBlockContainer<T> blk(data);
//     Store(key, blk);
//   }
//   /**
//    * \brief push a shared rowblock container into the store (no memory copy)
//    *
//    * @param key the unique key
//    * @param data the rowblock container
//    */
//   template <typename T>
//   void Store(const std::string& key, const SharedRowBlockContainer<T>& data) {
//     rowblk_keys_.insert(key);
//     CHECK_EQ(data.offset[0], 0);
//     auto keys = GetRowBlockKeys(key);
//     Store(keys[0], data.offset);
//     Store(keys[1], data.label);
//     Store(keys[2], data.weight);
//     Store(keys[3], data.index);
//     Store(keys[4], data.value);
//   }
//   /**
//    * \brief pull rowblock from the store
//    *
//    * @param key the unique key
//    * @param range an optional row range for pulling
//    * @param data the pulled data
//    */
//   template <typename T>
//   void Fetch(const std::string& key, SharedRowBlockContainer<T>* data,
//             Range range = Range::All()) {
//     CHECK_NOTNULL(data);
//     CHECK(IsRowBlockKey(key));
//     auto keys = GetRowBlockKeys(key);
//     Range rg1 = range == Range::All() ? range
//                 : Range(range.begin, range.end+1);
//     Fetch(keys[0], &data->offset, rg1);
//     Fetch(keys[1], &data->label, range);
//     Fetch(keys[2], &data->weight, range);
//     Range rg3 = Range(data->offset[0], data->offset.back());
//     Fetch(keys[3], &data->index, rg3);
//     Fetch(keys[4], &data->value, rg3);
//     if (rg3.begin != 0) {
//       SArray<size_t> offset; offset.CopyFrom(data->offset);
//       for (size_t& o : offset) o -= rg3.begin;
//       data->offset = offset;
//     }

//   }

// #include "dmlc/data.h"



// #include "reader/batch_reader.h"
// TEST(DataStore, RowBlock) {
//   BatchReader reader("../tests/data", "libsvm", 0, 1, 100);
//   CHECK(reader.Next());
//   auto data = reader.Value();

//   DataStore store;
//   store.Store("1", data);

//   SharedRowBlockContainer<feaid_t> blk1;
//   store.Fetch("1", &blk1);
//   check_equal(data, blk1.GetBlock());

//   SharedRowBlockContainer<feaid_t> blk2;
//   store.Fetch("1", &blk2, Range(10, 40));
//   check_equal(data.Slice(10, 40), blk2.GetBlock());
// }
