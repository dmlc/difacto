#include <gtest/gtest.h>
#include "./utils.h"
#include "reader/batch_reader.h"
#include "difacto/base.h"
#include "data/localizer.h"

using namespace difacto;

TEST(Localizer, Base) {
  BatchReader reader("../tests/data", "libsvm", 0, 1, 100);
  CHECK(reader.Next());
  dmlc::data::RowBlockContainer<unsigned> compact;
  std::vector<feaid_t> uidx;
  std::vector<real_t> freq;

  Localizer lc;
  lc.Compact(reader.Value(), &compact, &uidx, &freq);
  auto batch = compact.GetBlock();
  int size = batch.size;

  for (auto& i : uidx) i = ReverseBytes(i);

  EXPECT_EQ(norm1(uidx.data(), uidx.size()), (uint32_t)65111856);
  EXPECT_EQ(norm1(freq.data(), freq.size()), (real_t)9648);
  EXPECT_EQ(norm1(reader.Value().offset, size+1),
            norm1(batch.offset, size+1));
  EXPECT_EQ(norm2(reader.Value().value, batch.offset[size]),
            norm2(batch.value, batch.offset[size]));
}

TEST(Localizer, BaseHash) {
  BatchReader reader("../tests/data", "libsvm", 0, 1, 100);
  CHECK(reader.Next());
  dmlc::data::RowBlockContainer<unsigned> compact;
  std::vector<feaid_t> uidx;
  std::vector<real_t> freq;

  Localizer lc(1000);
  lc.Compact(reader.Value(), &compact, &uidx, &freq);
  auto batch = compact.GetBlock();
  int size = batch.size;

  for (auto& i : uidx) i = ReverseBytes(i);

  EXPECT_EQ(norm1(uidx.data(), uidx.size()), (uint32_t)478817);
  EXPECT_EQ(norm1(freq.data(), freq.size()), 9648);
  EXPECT_EQ(norm1(reader.Value().offset, size+1),
            norm1(batch.offset, size+1));
  EXPECT_EQ(norm2(reader.Value().value, batch.offset[size]),
            norm2(batch.value, batch.offset[size]));
}

TEST(Localizer, ReverseBytes) {
  feaid_t max = -1;
  int n = 1000000;
  for (int i = 0; i < n; ++i) {
    feaid_t j = (max / n) * i;
    EXPECT_EQ(j, ReverseBytes(ReverseBytes(j)));
  }
}
