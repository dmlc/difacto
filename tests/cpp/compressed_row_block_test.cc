#include <gtest/gtest.h>
#include "data/compressed_row_block.h"
#include "reader/batch_reader.h"
#include "./utils.h"

using namespace difacto;

TEST(CompressedRowBlock, Basic) {
  BatchReader reader("../tests/data", "libsvm", 0, 1, 100);
  CHECK(reader.Next());
  auto A = reader.Value();

  std::string out;
  CompressedRowBlock crb;
  crb.Compress(A, &out);

  dmlc::data::RowBlockContainer<feaid_t> container;
  crb.Decompress(out, &container);
  auto B = container.GetBlock();

  check_equal(A, B);
}
