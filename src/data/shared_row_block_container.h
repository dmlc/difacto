#ifndef DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#define DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#include <vector>
#include "data/row_block.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
namespace difacto {

/**
 * \brief a shared data structure holds a data block
 * \tparam IndexType the type of index we are using
 */
template<typename IndexType>
struct SharedRowBlockContainer {
  /** \brief default constructor */
  SharedRowBlockContainer() { }
  /**
   * \brief construct by copying from a rowblock
   * \param blk the rowblock
   */
  SharedRowBlockContainer(const dmlc::RowBlock<IndexType>& blk) {
    offset.CopyFrom(blk.offset, blk.size+1);
    if (blk.label != nullptr) {
      label.CopyFrom(blk.label, blk.size);
    }
    if (blk.weight != nullptr) {
      weight.CopyFrom(blk.weight, blk.size);
    }
    size_t nnz = blk.offset[blk.size] - blk.offset[0];
    index.CopyFrom(blk.index, nnz);
    if (blk.value != nullptr) {
      value.CopyFrom(blk.value, nnz);
    }
  }
  /**
   * \brief construct by moving from a rowblockcontainer
   *
   * CAUTION: advanced usage only. blk will be set nullptr after. If you only
   * want to copy the data, use the above copying constructor
   *
   * \param blk the rowblock
   */
  SharedRowBlockContainer(dmlc::data::RowBlockContainer<IndexType>** blk) {
    CHECK_NOTNULL(blk);
    CHECK_NOTNULL(*blk);
    std::shared_ptr<dmlc::data::RowBlockContainer<IndexType>> data(*blk);
    *blk = nullptr;

    // steal the data
    offset.reset(data->offset.data(), data->offset.size(),
                 [data](size_t* ptr) { data->offset.clear(); });
    label.reset(data->label.data(), data->label.size(),
                [data](real_t* ptr) { data->label.clear(); });
    weight.reset(data->weight.data(), data->weight.size(),
                 [data](real_t* ptr) { data->weight.clear(); });
    index.reset(data->index.data(), data->index.size(),
                [data](IndexType* ptr) { data->index.clear(); });
    value.reset(data->value.data(), data->value.size(),
                [data](real_t* ptr) { data->value.clear(); });
  }

  /*! \brief convert to a row block */
  dmlc::RowBlock<IndexType> GetBlock() const {
    dmlc::RowBlock<IndexType> blk;
    blk.size = offset.size() - 1;
    blk.offset = offset.data();
    blk.label = nullptr;
    if (label.size()) {
      // CHECK_EQ(blk.size, label.size());
      blk.label = label.data();
    }
    blk.weight = nullptr;
    if (weight.size()) {
      // CHECK_EQ(blk.size, weight.size());
      blk.weight = weight.data();
    }
    CHECK_EQ(index.size(), offset.back() - offset.front());
    blk.index = index.data();
    blk.value = nullptr;
    if (value.size()) {
      CHECK_EQ(value.size(), offset.back() - offset.front());
      blk.value = value.data();
    }
    return blk;
  }

  /*! \brief array[size+1], row pointer to beginning of each rows */
  SArray<size_t> offset;
  /*! \brief array[size] label of each instance */
  SArray<real_t> label;
  /*! \brief array[size] weight of each instance */
  SArray<real_t> weight;
  /*! \brief feature index */
  SArray<IndexType> index;
  /*! \brief feature value */
  SArray<real_t> value;
};

}  // namespace difacto
#endif  // DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
