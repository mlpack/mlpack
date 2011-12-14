/**
 * @file emst_test.cpp
 *
 * Test file for EMST methods.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::emst;

BOOST_AUTO_TEST_SUITE(EMSTTest);

/**
 * Simple emst test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  The dataset is
 * in one dimension for simplicity -- the correct functionality of distance
 * functions is not tested here.
 */
BOOST_AUTO_TEST_CASE(exhaustive_synthetic_test)
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

  // Now perform the actual calculation.
  arma::mat results;

  DualTreeBoruvka<> dtb(data);
  dtb.ComputeMST(results);

  // Now the exhaustive check for correctness.
  BOOST_REQUIRE(results(0, 0) == 1);
  BOOST_REQUIRE(results(1, 0) == 8);
  BOOST_REQUIRE_CLOSE(results(2, 0), 0.08, 1e-5);

  BOOST_REQUIRE(results(0, 1) == 9);
  BOOST_REQUIRE(results(1, 1) == 10);
  BOOST_REQUIRE_CLOSE(results(2, 1), 0.09, 1e-5);

  BOOST_REQUIRE(results(0, 2) == 0);
  BOOST_REQUIRE(results(1, 2) == 2);
  BOOST_REQUIRE_CLOSE(results(2, 2), 0.1, 1e-5);

  BOOST_REQUIRE(results(0, 3) == 1);
  BOOST_REQUIRE(results(1, 3) == 2);
  BOOST_REQUIRE_CLOSE(results(2, 3), 0.22, 1e-5);

  BOOST_REQUIRE(results(0, 4) == 3);
  BOOST_REQUIRE(results(1, 4) == 10);
  BOOST_REQUIRE_CLOSE(results(2, 4), 0.25, 1e-5);

  BOOST_REQUIRE(results(0, 5) == 0);
  BOOST_REQUIRE(results(1, 5) == 5);
  BOOST_REQUIRE_CLOSE(results(2, 5), 0.27, 1e-5);

  BOOST_REQUIRE(results(0, 6) == 8);
  BOOST_REQUIRE(results(1, 6) == 9);
  BOOST_REQUIRE_CLOSE(results(2, 6), 0.46, 1e-5);

  BOOST_REQUIRE(results(0, 7) == 6);
  BOOST_REQUIRE(results(1, 7) == 7);
  BOOST_REQUIRE_CLOSE(results(2, 7), 0.7, 1e-5);

  BOOST_REQUIRE(results(0, 8) == 5);
  BOOST_REQUIRE(results(1, 8) == 7);
  BOOST_REQUIRE_CLOSE(results(2, 8), 1.08, 1e-5);

  BOOST_REQUIRE(results(0, 9) == 3);
  BOOST_REQUIRE(results(1, 9) == 4);
  BOOST_REQUIRE_CLOSE(results(2, 9), 3.8, 1e-5);
}

/**
 * Test the dual tree method against the naive computation.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(dual_tree_vs_naive)
{
  arma::mat input_data;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (!data::Load("test_data_3_1000.csv", input_data))
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat dual_data = arma::trans(input_data);
  arma::mat naive_data = arma::trans(input_data);

  // Reset parameters from last test.
  DualTreeBoruvka<> dtb(dual_data);

  arma::mat dual_results;
  dtb.ComputeMST(dual_results);

  // Set naive mode.
  DualTreeBoruvka<> dtb_naive(naive_data, true);

  arma::mat naive_results;
  dtb_naive.ComputeMST(naive_results);

  BOOST_REQUIRE(dual_results.n_cols == naive_results.n_cols);
  BOOST_REQUIRE(dual_results.n_rows == naive_results.n_rows);

  for (size_t i = 0; i < dual_results.n_cols; i++)
  {
    BOOST_REQUIRE(dual_results(0, i) == naive_results(0, i));
    BOOST_REQUIRE(dual_results(1, i) == naive_results(1, i));
    BOOST_REQUIRE_CLOSE(dual_results(2, i), naive_results(2, i), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
