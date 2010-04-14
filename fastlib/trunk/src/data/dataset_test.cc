/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
#include "dataset.h"
//#include "dataset.h"

#include "../math/discrete.h"
#include "../base/test.h"

TEST_SUITE_BEGIN(dataset)

void TestSplitTrainTest() {
  Dataset orig;
  
  orig.InitBlank();
  orig.matrix().Init(1, 12);
  orig.info().InitContinuous(1);
  
  for (int i = 0; i < 12; i++) {
    orig.matrix().set(0, i, i);
  }
  
  ArrayList<int> found;
  
  found.Init(12);
  
  for (int i = 0; i < 12; i++) {
    found[i] = 0;
  }
  
  Dataset train;
  Dataset test;
  
  ArrayList<index_t> permutation;
  
  math::MakeIdentityPermutation(12, &permutation);
  
  orig.SplitTrainTest(5, 1, permutation,
      &train, &test);
  
  DEBUG_ASSERT(test.n_points() == 3);
  DEBUG_ASSERT(train.n_points() == 9);
  DEBUG_ASSERT_MSG(test.get(0, 0) == 1, "%f", (test.get(0, 0)));
  DEBUG_ASSERT_MSG(test.get(0, 1) == 6, "%f", (test.get(0, 1)));
  DEBUG_ASSERT_MSG(test.get(0, 2) == 11, "%f", (test.get(0, 2)));
  
  DEBUG_ASSERT_MSG(train.get(0, 0) == 0, "%f", (train.get(0, 0)));
  DEBUG_ASSERT_MSG(train.get(0, 1) == 2, "%f", (train.get(0, 1)));
  DEBUG_ASSERT_MSG(train.get(0, 2) == 3, "%f", (train.get(0, 2)));
  DEBUG_ASSERT_MSG(train.get(0, 3) == 4, "%f", (train.get(0, 3)));
  DEBUG_ASSERT_MSG(train.get(0, 4) == 5, "%f", (train.get(0, 4)));
  DEBUG_ASSERT_MSG(train.get(0, 5) == 7, "%f", (train.get(0, 5)));
  DEBUG_ASSERT_MSG(train.get(0, 6) == 8, "%f", (train.get(0, 6)));
  DEBUG_ASSERT_MSG(train.get(0, 7) == 9, "%f", (train.get(0, 7)));
  DEBUG_ASSERT_MSG(train.get(0, 8) == 10, "%f", (train.get(0, 8)));
}

void AssertSameMatrix(const Matrix& a, const Matrix& b) {
  index_t r = a.n_rows();
  index_t c = a.n_cols();
  
  TEST_ASSERT(a.n_rows() == b.n_rows());
  TEST_ASSERT(a.n_cols() == b.n_cols());
  
  for (index_t ri = 0; ri < r; ri++) {
    for (index_t ci = 0; ci < c; ci++) {
      DEBUG_ASSERT_MSG(a.get(ri, ci) == b.get(ri, ci), "(%d, %d): %f != %f",
          ri, ci, a.get(ri, ci), b.get(ri, ci));
    }
  }
}

void TestLoad() {
  Dataset d1;
  Dataset d2;
  Dataset d3;
  Dataset d4;
  Dataset d5;
  
  MUST_PASS(d1.InitFromFile("test/fake.arff"));
  MUST_PASS(d2.InitFromFile("test/fake.csv"));
  MUST_PASS(d3.InitFromFile("test/fake.csvh"));
  MUST_PASS(d4.InitFromFile("test/fake.tsv"));
  MUST_PASS(d5.InitFromFile("test/fake.weird"));
  
  AssertSameMatrix(d1.matrix(), d2.matrix());
  AssertSameMatrix(d1.matrix(), d3.matrix());
  AssertSameMatrix(d1.matrix(), d4.matrix());
  AssertSameMatrix(d1.matrix(), d5.matrix());
}

void TestStoreLoad() {
  Dataset d1;
  Dataset d2;
  Dataset d3;
  
  MUST_PASS(d1.InitFromFile("test/fake.arff"));
  d1.WriteCsv("test/fakeout1.csv");
  d1.WriteArff("test/fakeout1.arff");
  
  MUST_PASS(d2.InitFromFile("test/fakeout1.arff"));
  MUST_PASS(d3.InitFromFile("test/fakeout1.csv"));
  
  AssertSameMatrix(d1.matrix(), d2.matrix());
  AssertSameMatrix(d1.matrix(), d3.matrix());
  
  DEBUG_ASSERT_MSG(strcmp(d1.info().name(), d2.info().name()) == 0,
      "%s != %s", d1.info().name(), d2.info().name());
  for (index_t i = 0; i < d1.info().n_features(); i++) {
    DEBUG_ASSERT(
      strcmp(d1.info().feature(i).name(), d2.info().feature(i).name()) == 0);
    DEBUG_ASSERT(d1.info().feature(i).type() == d2.info().feature(i).type());
  }
}

TEST_SUITE_END(dataset, TestSplitTrainTest, TestLoad, TestStoreLoad)

/*
int main(int argc, char *argv[]) {
  xrun_init(argc, argv);
  const char *in = xrun_param_str("in");
  const char *out = xrun_param_str("out");
  String type;
  
  type.Copy(xrun_param_str("type"));
  
  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(in))) return 1;
  
  success_t result;
  
  if (type.EqualsNoCase("arff")) {
    result = dataset.WriteArff(out);
  } else if (type.EqualsNoCase("csv")) {
    result = dataset.WriteCsv(out, false);
  } else if (type.EqualsNoCase("csvh")) {
    result = dataset.WriteCsv(out, true);
  } else {
    result = SUCCESS_FAIL;
  }
  
  if (!PASSED(result)) {
    fprintf(stderr, "Error!\n");
    return 1;
  }
  
  ArrayList<index_t> permutation;
  math::MakeRandomPermutation(dataset.n_points(), &permutation);
  
  for (int k = 5; k < 10; k++) {
    Dataset test;
    Dataset train;
    int i = k - 5;
    dataset.SplitTrainTest(k, i, permutation, &train, &test);
    assert(test.n_points() + train.n_points() == dataset.n_points());
  }

  return 0;
}
*/
