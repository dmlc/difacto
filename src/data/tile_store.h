/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_DATA_TILE_STORE_H_
#define DIFACTO_DATA_TILE_STORE_H_
#include <string>
#include <vector>
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
  TileStore() { }
  ~TileStore() { delete data_; }
  friend class TileBuilder;

  KWArgs Init(const KWArgs& kwargs) {
    data_ = new DataStore();
    return kwargs;
  }
  /**
   * \brief store a shared rowblock container into the store (no memory copy)
   * @param rowblk_id rowblock id
   * @param data the rowblock container
   */
  void Store(int rowblk_id,
             const SharedRowBlockContainer<unsigned>& data) {
    auto key = std::to_string(rowblk_id) + "_";
    data_->Store(key+"label", data.label);
    data_->Store(key+"offset", data.offset);
    data_->Store(key+"index", data.index);
    data_->Store(key+"value", data.value);
  }

  /**
   * \brief store a the column map of a row block into the store (no memory copy)
   */
  void Store(int rowblk_id, const SArray<int>& colmap) {
    data_->Store(std::to_string(rowblk_id) + "_colmap", colmap);
  }

  /**
   * \brief prefetch a tile
   *
   * @param rowblk_id
   * @param colblk_id
   */
  void Prefetch(int rowblk_id, int colblk_id) {
    auto key = std::to_string(rowblk_id) + "_";
    auto rg = meta_[rowblk_id][colblk_id];
    data_->Prefetch(key+"label");
    data_->Prefetch(key+"colmap", rg.colmap);
    data_->Prefetch(key+"offset", rg.offset);
    data_->Prefetch(key+"index", rg.index);
    data_->Prefetch(key+"value", rg.index);
  }

  /**
   * \brief fetch a tile
   *
   * @param rowblk_id
   * @param colblk_id
   * @param tile
   */
  void Fetch(int rowblk_id, int colblk_id, Tile* tile) {
    auto& data = CHECK_NOTNULL(tile)->data;
    auto key = std::to_string(rowblk_id) + "_";
    auto rg = meta_[rowblk_id][colblk_id];
    data_->Fetch(key+"label", &data.label);
    data_->Fetch(key+"colmap", &tile->colmap, rg.colmap);
    data_->Fetch(key+"offset", &data.offset, rg.offset);
    if (rg.offset.begin != 0) {
      // force to start from 0
      SArray<size_t> offset; offset.CopyFrom(data.offset);
      for (size_t i = 0; i < offset.size(); ++i) {
        offset[i] -= data.offset[0];
      }
      data.offset = offset;
    }
    data_->Fetch(key+"index", &data.index, rg.index);
    data_->Fetch(key+"value", &data.value, rg.index);
  }

  /**
   * \brief load meta data
   */
  void Load(dmlc::Stream *fi) {
    dmlc::istream is(fi);
    std::string header; is >> header;
    CHECK_EQ(header, meta_header_) << "invalid meta header";
    size_t size; is >> size; meta_.resize(size);
    std::string format; is >> format;
    CHECK_EQ(format, meta_format_) << "invalid meta header";
    for (size_t i = 0; i < size; ++i) {
      size_t col_size; is >> col_size;
      for (size_t j = 0; j < col_size; ++j) {
        Meta c; is >> c.colmap.begin >> c.colmap.end
                   >> c.offset.begin >> c.offset.end
                   >> c.index.begin >> c.index.end;
        meta_[i].push_back(c);
      }
    }
  }

  /**
   * \brief save meta data
   */
  void Save(dmlc::Stream *fo) const {
    dmlc::ostream os(fo);
    os << meta_header_ << "\t" << meta_.size() << "\n" << meta_format_ << "\n";
    for (size_t i = 0; i < meta_.size(); ++i) {
      os << meta_[i].size();
      for (auto c : meta_[i]) {
        os << "\t" << c.colmap.begin << "\t" << c.colmap.end
           << "\t" << c.offset.begin << "\t" << c.offset.end
           << "\t" << c.index.begin << "\t" << c.index.end;
      }
      os << "\n";
    }
  }

 private:
  DataStore* data_ = nullptr;
  /** \brief meta data for a rowblk */
  struct Meta { Range colmap; Range offset; Range index; };
  std::vector<std::vector<Meta>> meta_;  // row x col

  const std::string meta_header_ = "tile_store_meta";
  const std::string meta_format_ =
    "format:transposed,colblk_size,colmap_range,offset_range,index_range,...";
};

}  // namespace difacto
#endif  // DIFACTO_DATA_TILE_STORE_H_
