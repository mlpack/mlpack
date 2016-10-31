/**
 * @file emst_test.cpp
 *
 * Test file for EMST methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <mlpack/core/tree/cover_tree.hpp>

using namespace mlpack;
using namespace mlpack::emst;
using namespace mlpack::tree;
using namespace mlpack::bound;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(EMSTTest);

/**
 * Simple emst test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, naive) produces the correct results.  The dataset is in one
 * dimension for simplicity -- the correct functionality of distance functions
 * is not tested here.
 */
BOOST_AUTO_TEST_CASE(ExhaustiveSyntheticTest)
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
    BOOST_REQUIRE_EQUAL(results(0, 0), newFromOld[1]);
    BOOST_REQUIRE_EQUAL(results(1, 0), newFromOld[8]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 0), newFromOld[1]);
    BOOST_REQUIRE_EQUAL(results(0, 0), newFromOld[8]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 0), 0.08, 1e-5);

  if (newFromOld[9] < newFromOld[10])
  {
    BOOST_REQUIRE_EQUAL(results(0, 1), newFromOld[9]);
    BOOST_REQUIRE_EQUAL(results(1, 1), newFromOld[10]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 1), newFromOld[9]);
    BOOST_REQUIRE_EQUAL(results(0, 1), newFromOld[10]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 1), 0.09, 1e-5);

  if (newFromOld[0] < newFromOld[2])
  {
    BOOST_REQUIRE_EQUAL(results(0, 2), newFromOld[0]);
    BOOST_REQUIRE_EQUAL(results(1, 2), newFromOld[2]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 2), newFromOld[0]);
    BOOST_REQUIRE_EQUAL(results(0, 2), newFromOld[2]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 2), 0.1, 1e-5);

  if (newFromOld[1] < newFromOld[2])
  {
    BOOST_REQUIRE_EQUAL(results(0, 3), newFromOld[1]);
    BOOST_REQUIRE_EQUAL(results(1, 3), newFromOld[2]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 3), newFromOld[1]);
    BOOST_REQUIRE_EQUAL(results(0, 3), newFromOld[2]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 3), 0.22, 1e-5);

  if (newFromOld[3] < newFromOld[10])
  {
    BOOST_REQUIRE_EQUAL(results(0, 4), newFromOld[3]);
    BOOST_REQUIRE_EQUAL(results(1, 4), newFromOld[10]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 4), newFromOld[3]);
    BOOST_REQUIRE_EQUAL(results(0, 4), newFromOld[10]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 4), 0.25, 1e-5);

  if (newFromOld[0] < newFromOld[5])
  {
    BOOST_REQUIRE_EQUAL(results(0, 5), newFromOld[0]);
    BOOST_REQUIRE_EQUAL(results(1, 5), newFromOld[5]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 5), newFromOld[0]);
    BOOST_REQUIRE_EQUAL(results(0, 5), newFromOld[5]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 5), 0.27, 1e-5);

  if (newFromOld[8] < newFromOld[9])
  {
    BOOST_REQUIRE_EQUAL(results(0, 6), newFromOld[8]);
    BOOST_REQUIRE_EQUAL(results(1, 6), newFromOld[9]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 6), newFromOld[8]);
    BOOST_REQUIRE_EQUAL(results(0, 6), newFromOld[9]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 6), 0.46, 1e-5);

  if (newFromOld[6] < newFromOld[7])
  {
    BOOST_REQUIRE_EQUAL(results(0, 7), newFromOld[6]);
    BOOST_REQUIRE_EQUAL(results(1, 7), newFromOld[7]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 7), newFromOld[6]);
    BOOST_REQUIRE_EQUAL(results(0, 7), newFromOld[7]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 7), 0.7, 1e-5);

  if (newFromOld[5] < newFromOld[7])
  {
    BOOST_REQUIRE_EQUAL(results(0, 8), newFromOld[5]);
    BOOST_REQUIRE_EQUAL(results(1, 8), newFromOld[7]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 8), newFromOld[5]);
    BOOST_REQUIRE_EQUAL(results(0, 8), newFromOld[7]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 8), 1.08, 1e-5);

  if (newFromOld[3] < newFromOld[4])
  {
    BOOST_REQUIRE_EQUAL(results(0, 9), newFromOld[3]);
    BOOST_REQUIRE_EQUAL(results(1, 9), newFromOld[4]);
  }
  else
  {
    BOOST_REQUIRE_EQUAL(results(1, 9), newFromOld[3]);
    BOOST_REQUIRE_EQUAL(results(0, 9), newFromOld[4]);
  }
  BOOST_REQUIRE_CLOSE(results(2, 9), 3.8, 1e-5);
}

/**
 * Test the dual tree method against the naive computation.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive)
{
  arma::mat inputData;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", inputData))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

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

  BOOST_REQUIRE_EQUAL(dualResults.n_cols, naiveResults.n_cols);
  BOOST_REQUIRE_EQUAL(dualResults.n_rows, naiveResults.n_rows);

  for (size_t i = 0; i < dualResults.n_cols; i++)
  {
    BOOST_REQUIRE_EQUAL(dualResults(0, i), naiveResults(0, i));
    BOOST_REQUIRE_EQUAL(dualResults(1, i), naiveResults(1, i));
    BOOST_REQUIRE_CLOSE(dualResults(2, i), naiveResults(2, i), 1e-5);
  }
}

/**
 * Make sure the cover tree works fine.
 */
BOOST_AUTO_TEST_CASE(CoverTreeTest)
{
  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  DualTreeBoruvka<> bst(inputData);
  DualTreeBoruvka<EuclideanDistance, arma::mat, StandardCoverTree>
      ct(inputData);

  arma::mat bstResults;
  arma::mat coverResults;

  // Run the algorithms.
  bst.ComputeMST(bstResults);
  ct.ComputeMST(coverResults);

  for (size_t i = 0; i < bstResults.n_cols; i++)
  {
    BOOST_REQUIRE_EQUAL(bstResults(0, i), coverResults(0, i));
    BOOST_REQUIRE_EQUAL(bstResults(1, i), coverResults(1, i));
    BOOST_REQUIRE_CLOSE(bstResults(2, i), coverResults(2, i), 1e-5);
  }

}

/**
 * Test BinarySpaceTree with Ball Bound.
 */
BOOST_AUTO_TEST_CASE(BallTreeTest)
{
  arma::mat inputData;
  if (!data::Load("test_data_3_1000.csv", inputData))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // naive mode.
  DualTreeBoruvka<> bst(inputData, true);
  // Ball tree.
  DualTreeBoruvka<EuclideanDistance, arma::mat, BallTree> ballt(inputData);

  arma::mat bstResults;
  arma::mat ballResults;

  // Run the algorithms.
  bst.ComputeMST(bstResults);
  ballt.ComputeMST(ballResults);

  for (size_t i = 0; i < bstResults.n_cols; i++)
  {
    BOOST_REQUIRE_EQUAL(bstResults(0, i), ballResults(0, i));
    BOOST_REQUIRE_EQUAL(bstResults(1, i), ballResults(1, i));
    BOOST_REQUIRE_CLOSE(bstResults(2, i), ballResults(2, i), 1e-5);
  }

}

BOOST_AUTO_TEST_SUITE_END();
