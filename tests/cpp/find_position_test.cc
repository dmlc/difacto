#include <gtest/gtest.h>
#include <unordered_map>
#include "./utils.h"
#include "common/find_position.h"

using namespace difacto;

TEST(FindPosition, Basic) {
  SArray<int> a = {3, 5, 7};
  SArray<int> b = {1, 3, 4, 7, 8};
  SArray<int> pos, pos2 = {-1, 0, -1, 2, -1};
  FindPosition(a, b, &pos);
  for (size_t i = 0; i < pos2.size(); ++i) EXPECT_EQ(pos2[i], pos[i]);

  SArray<int> pos3, pos4 = {1, -1, 3};
  FindPosition(b, a, &pos3);
  for (size_t i = 0; i < pos4.size(); ++i) EXPECT_EQ(pos4[i], pos3[i]);
}
