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
  dmlc::RowBlock<unsigned> data;
};

class TileBuilder;

class TileStore {
 public:
  friend class TileBuilder;

  void Prefetch(int rowblk_id, int colblk_id) {

  }
  void Fetch(int rowblk_id, int colblk_id, Tile* tile) {

  }



  void StoreColmap(int rowblk_id, const SArray<int>& colmap) {

  }


 private:
  DataStore* data_;
  std::vector<std::vector<Range>> colblk_pos_;
};

}  // namespace bcd
}  // namespace difacto
#endif  // _TILE_STORE_H_

      // Range fea_blk_pos = feablk_pos_[d+1][fea_blk_id];
      // auto id = std::to_string(d) + "_";
      // SArray<int> fea_map;
      // data_store_->Fetch(id + "feamap", &fea_map, fea_blk_pos);

      // SArray<int> blk_grad_len(fea_map.size());
      // for (size_t i = 0; i < fea_map.size(); ++i) {
      //   blk_grad_len[i] = grad_len[fea_map[i]];
      // }

      // SharedRowBlockContainer<unsigned> data;
      // data_store_->Fetch(id + "data", &data, fea_blk_pos);

      // LogitLossDelta* loss = new LogitLossDelta();
      // // loss->Init();
      // SArray<real_t> pred;
      // data_store_->Fetch(id + "predict", &pred);

      // // calc grad
      // SArray<real_t> blk_grad;
      // loss->CalcGrad(data.GetBlock(),
      //                {SArray<char>(pred), SArray<char>(blk_grad_len)},
      //                &blk_grad);
