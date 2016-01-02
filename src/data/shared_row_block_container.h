#ifndef DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#define DIFACTO_DATA_SHARED_ROW_BLOCK_CONTAINER_H_
#include <vector>
#include "data/row_block.h"
#include "difacto/base.h"
#include "./sarray.h"
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
   * \brief construct from a rowblockcontainer
   * \param blk the user should not delete rowblk
   */
  SharedRowBlockContainer(dmlc::data::RowBlockContainer<IndexType>* blk) {

  }

  /*! \brief convert to a row block */
  dmlc::RowBlock<IndexType> GetBlock() const {

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
