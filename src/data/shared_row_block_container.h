#ifndef DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#define DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#include <vector>
#include "data/row_block.h"
#include "difacto/base.h"
#include "difacto/sarray.h"
namespace difacto {

/**
 * \brief a shared data structure holds a row block
 * \tparam IndexType the type of index we are using
 */
template<typename IndexType>
struct SharedRowBlockContainer {
  /** \brief default constructor */
  SharedRowBlockContainer() { }
  /**
   * \brief construct by moving from a rowblockcontainer
   * \param blk the user should NOT delete blk
   */
  SharedRowBlockContainer(dmlc::data::RowBlockContainer<IndexType>* blk) {
    // TODO
  }

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

  /*! \brief convert to a row block */
  dmlc::RowBlock<IndexType> GetBlock() const {
    // TODO
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
