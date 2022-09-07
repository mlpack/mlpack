/**
 * @file tests/emst_test.cpp
 *
 * Test file for EMST methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/emst.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * Simple emst test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, naive) produces the correct results.  The dataset is in one
 * dimension for simplicity -- the correct functionality of distance functions
 * is not tested here.
 */
TEST_CASE("EMSTExhaustiveSyntheticTest", "[EMSTTest]")
{
  // Set up our data.
  arma::mat data(1, 11);
  data[0] = 0.05; // Row addressing is unnecessary (they are all 0).
  data[1] = 0.37;
  data[2] = 0.15;
  data[3] = 1.25;
  data[4] = 5.05;
  data[5] = -0.22;
  data[6] = -2.00;
  data[7] = -1.30;
  data[8] = 0.45;
  data[9] = 0.91;
  data[10] = 1.00;

  arma::mat results;

  // Build the tree by hand to get a leaf size of 1.
  typedef KDTree<EuclideanDistance, DTBStat, arma::mat> TreeType;
  std::vector<size_t> oldFromNew;
  std::vector<size_t> newFromOld;
  TreeType tree(data, oldFromNew, newFromOld, 1);

  // Create the DTB object and run the calculation.
  DualTreeBoruvka<> dtb(&tree);
  dtb.ComputeMST(results);

  // Now the exhaustive check for correctness.
  if (newFromOld[1] < newFromOld[8])
  {
    REQUIRE(results(0, 0) == newFromOld[1]);
    REQUIRE(results(1, 0) == newFromOld[8]);
  }
  else
  {
    REQUIRE(results(1, 0) == newFromOld[1]);
    REQUIRE(results(0, 0) == newFromOld[8]);
  }
  REQUIRE(results(2, 0) == Approx(0.08).epsilon(1e-7));

  if (newFromOld[9] < newFromOld[10])
  {
    REQUIRE(results(0, 1) == newFromOld[9]);
    REQUIRE(results(1, 1) == newFromOld[10]);
  }
  else
  {
    REQUIRE(results(1, 1) == newFromOld[9]);
    REQUIRE(results(0, 1) == newFromOld[10]);
  }
  REQUIRE(results(2, 1) == Approx(0.09).epsilon(1e-7));

  if (newFromOld[0] < newFromOld[2])
  {
    REQUIRE(results(0, 2) == newFromOld[0]);
    REQUIRE(results(1, 2) == newFromOld[2]);
  }
  else
  {
    REQUIRE(results(1, 2) == newFromOld[0]);
    REQUIRE(results(0, 2) == newFromOld[2]);
  }
  REQUIRE(results(2, 2) == Approx(0.1).epsilon(1e-7));

  if (newFromOld[1] < newFromOld[2])
  {
    REQUIRE(results(0, 3) == newFromOld[1]);
    REQUIRE(results(1, 3) == newFromOld[2]);
  }
  else
  {
    REQUIRE(results(1, 3) == newFromOld[1]);
    REQUIRE(results(0, 3) == newFromOld[2]);
  }
  REQUIRE(results(2, 3) == Approx(0.22).epsilon(1e-7));

  if (newFromOld[3] < newFromOld[10])
  {
    REQUIRE(results(0, 4) == newFromOld[3]);
    REQUIRE(results(1, 4) == newFromOld[10]);
  }
  else
  {
    REQUIRE(results(1, 4) == newFromOld[3]);
    REQUIRE(results(0, 4) == newFromOld[10]);
  }
  REQUIRE(results(2, 4) == Approx(0.25).epsilon(1e-7));

  if (newFromOld[0] < newFromOld[5])
  {
    REQUIRE(results(0, 5) == newFromOld[0]);
    REQUIRE(results(1, 5) == newFromOld[5]);
  }
  else
  {
    REQUIRE(results(1, 5) == newFromOld[0]);
    REQUIRE(results(0, 5) == newFromOld[5]);
  }
  REQUIRE(results(2, 5) == Approx(0.27).epsilon(1e-7));

  if (newFromOld[8] < newFromOld[9])
  {
    REQUIRE(results(0, 6) == newFromOld[8]);
    REQUIRE(results(1, 6) == newFromOld[9]);
  }
  else
  {
    REQUIRE(results(1, 6) == newFromOld[8]);
    REQUIRE(results(0, 6) == newFromOld[9]);
  }
  REQUIRE(results(2, 6) == Approx(0.46).epsilon(1e-7));

  if (newFromOld[6] < newFromOld[7])
  {
    REQUIRE(results(0, 7) == newFromOld[6]);
    REQUIRE(results(1, 7) == newFromOld[7]);
  }
  else
  {
    REQUIRE(results(1, 7) == newFromOld[6]);
    REQUIRE(results(0, 7) == newFromOld[7]);
  }
  REQUIRE(results(2, 7) == Approx(0.7).epsilon(1e-7));

  if (newFromOld[5] < newFromOld[7])
  {
    REQUIRE(results(0, 8) == newFromOld[5]);
    REQUIRE(results(1, 8) == newFromOld[7]);
  }
  else
  {
    REQUIRE(results(1, 8) == newFromOld[5]);
    REQUIRE(results(0, 8) == newFromOld[7]);
  }
  REQUIRE(results(2, 8) == Approx(1.08).epsilon(1e-7));

  if (newFromOld[3] < newFromOld[4])
  {
    REQUIRE(results(0, 9) == newFromOld[3]);
    REQUIRE(results(1, 9) == newFromOld[4]);
  }
  else
  {
    REQUIRE(results(1, 9) == newFromOld[3]);
    REQUIRE(results(0, 9) == newFromOld[4]);
  }
  REQUIRE(results(2, 9) == Approx(3.8).epsilon(1e-7));
}

/**
 * Test the dual tree method against the naive computation.
 *
 * Errors are produced if the results are not identical.
 */
TEST_CASE("DualTreeVsNaive", "[EMSTTest]")
{
  arma::mat inputData;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", inputData))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dualData = inputData;
  arma::mat naiveData = inputData;

  // Reset parameters from last test.
  DualTreeBoruvka<> dtb(dualData);

  arma::mat dualResults;
  dtb.ComputeMST(dualResults);

  // Set naive mode.
  DualTreeBoruvka<> dtbNaive(naiveData, true);

  arma::mat naiveResults;
  dtbNaive.ComputeMST(naiveResults);

  REQUIRE(dualResults.n_cols == naiveResults.n_cols);
  REQUIRE(dualResults.n_rows == naiveResults.n_rows);

  for (size_t i = 0; i < dualResults.n_cols; ++i)
  {
    REQUIRE(dualResults(0, i) == naiveResults(0, i));
    REQUIRE(dualResults(1, i) == naiveResults(1, i));
    REQUIRE(dualResults(2, i) == Approx(naiveResults(2, i)).epsilon(1e-7));
  }
}

/**
 * Make sure the cover tree works fine.
 */
TEST_CASE("EMSTCoverTreeTest", "[EMSTTest]")
{
  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  DualTreeBoruvka<> bst(inputData);
  DualTreeBoruvka<EuclideanDistance, arma::mat, StandardCoverTree>
      ct(inputData);

  arma::mat bstResults;
  arma::mat coverResults;

  // Run the algorithms.
  bst.ComputeMST(bstResults);
  ct.ComputeMST(coverResults);

  for (size_t i = 0; i < bstResults.n_cols; ++i)
  {
    REQUIRE(bstResults(0, i) == coverResults(0, i));
    REQUIRE(bstResults(1, i) == coverResults(1, i));
    REQUIRE(bstResults(2, i) == Approx(coverResults(2, i)).epsilon(1e-7));
  }
}

/**
 * Test BinarySpaceTree with Ball Bound.
 */
TEST_CASE("EMSTBallTreeTest", "[EMSTTest]")
{
  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // naive mode.
  DualTreeBoruvka<> bst(inputData, true);
  // Ball tree.
  DualTreeBoruvka<EuclideanDistance, arma::mat, BallTree> ballt(inputData);

  arma::mat bstResults;
  arma::mat ballResults;

  // Run the algorithms.
  bst.ComputeMST(bstResults);
  ballt.ComputeMST(ballResults);

  for (size_t i = 0; i < bstResults.n_cols; ++i)
  {
    REQUIRE(bstResults(0, i) == ballResults(0, i));
    REQUIRE(bstResults(1, i) == ballResults(1, i));
    REQUIRE(bstResults(2, i) == Approx(ballResults(2, i)).epsilon(1e-7));
  }
}
