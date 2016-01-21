#ifndef _TILE_STORE_H_
#define _TILE_STORE_H_
#include "dmlc/data.h"
#include "difacto/sarray.h"
namespace difacto {
namespace bcd {

/**
 * \brief a row and column block X, stored in column-major order, namely X'
 */
struct Tile {
  SArray<int> colmap;
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
    data_->Prefetch(id + "_data", rg);
    data_->Prefetch(id + "_label");
    data_->Prefetch(id + "_colmap", rg);
  }

  void Fetch(int rowblk_id, int colblk_id, Tile* tile) {
    auto id = std::to_string(rowblk_id);
    auto rg = colblk_pos_[rowblk_id][colblk_id];
    CHECK_NOTNULL(tile);
    data_->Fetch(id + "_data", &tile->data, rg);
    data_->Fetch(id + "_colmap", &tile->colmap, rg);
    data_->Fetch(id + "_colmap", &tile->data.label);
  }

 private:
  DataStore* data_;
  std::vector<std::vector<Range>> colblk_pos_;
};

}  // namespace bcd
}  // namespace difacto
#endif  // _TILE_STORE_H_
