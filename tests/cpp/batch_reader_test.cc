/**
 *  Copyright (c) 2015 by Contributors
 */
#include <gtest/gtest.h>
#include "reader/batch_reader.h"
#include "./utils.h"

using namespace difacto;
int batch_size = 37;
int label[] = { 11,    15,   10};
int len[] = { 37,    37,    26};
size_t os[] = {     85035 ,      63968     ,  31323 };
uint32_t idx[] = {   95285478 ,   70504854  ,  62972349};
float val[] = {    37.0000 ,  37.0000 ,  26.0000};

TEST(BatchReader, Read) {
  BatchReader reader("../tests/data", "libsvm", 0, 1, batch_size);
  int i = 0;
  while (reader.Next()) {
    auto batch = reader.Value();
    int size = batch.size;
    EXPECT_EQ(label[i], sum(batch.label, size));
    EXPECT_EQ(len[i], size);
    EXPECT_EQ(os[i], norm1(batch.offset, size+1));
    EXPECT_EQ(idx[i], norm1(batch.index, batch.offset[size]));
    EXPECT_LE(fabs(val[i] - norm2(batch.value, batch.offset[size])), 1e-5);
    ++ i;
  }
}

TEST(BatchReader, RandRead) {
  BatchReader reader("../tests/data", "libsvm", 0, 1, batch_size, batch_size);
  int i = 0;
  while (reader.Next()) {
    auto batch = reader.Value();
    int size = batch.size;
    EXPECT_EQ(label[i], sum(batch.label, size));
    EXPECT_EQ(len[i], size);
    EXPECT_NE(os[i], norm1(batch.offset, size+1));
    EXPECT_EQ(idx[i], norm1(batch.index, batch.offset[size]));
    EXPECT_LE(fabs(val[i] - norm2(batch.value, batch.offset[size])), 1e-5);
    ++ i;
  }
}

TEST(BatchReader, PartRead) {
  BatchReader reader("../tests/data", "libsvm", 1, 2, batch_size);
  int ttl = 0;
  while (reader.Next()) {
    auto batch = reader.Value();
    int size = batch.size;
    EXPECT_LE(fabs(size - norm2(batch.value, batch.offset[size])), 1e-5);
    ttl += size;
  }
  CHECK_LE(ttl, 60);
  CHECK_GE(ttl, 40);
}
