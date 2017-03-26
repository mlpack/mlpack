/**
 * @file init_rules_test.cpp
 * @author Marcus Edel
 *
 * Tests for the various weight initialize methods.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/math/random.hpp>

#include <mlpack/methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/init_rules/oivs_init.hpp>
#include <mlpack/methods/ann/init_rules/orthogonal_init.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>


#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(InitRulesTest);

// Test the RandomInitialization class with a constant value.
BOOST_AUTO_TEST_CASE(ConstantInitTest)
{
  arma::mat weights;
  RandomInitialization constantInit(1, 1);
  constantInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 1);
  BOOST_REQUIRE_EQUAL(b, 1);
}

// Test the OrthogonalInitialization class.
BOOST_AUTO_TEST_CASE(OrthogonalInitTest)
{
  arma::mat weights;
  OrthogonalInitialization orthogonalInit;
  orthogonalInit.Initialize(weights, 100, 200);

  arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
  weights *= weights.t();

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);

  orthogonalInit.Initialize(weights, 200, 100);
  weights = weights.t() * weights;

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);
}

// Test the OrthogonalInitialization class with a non default gain.
BOOST_AUTO_TEST_CASE(OrthogonalInitGainTest)
{
  arma::mat weights;

  const double gain = 2;
  OrthogonalInitialization orthogonalInit(gain);
  orthogonalInit.Initialize(weights, 100, 200);

  arma::mat orthogonalWeights = arma::eye<arma::mat>(100, 100);
  orthogonalWeights *= (gain * gain);
  weights *= weights.t();

  for (size_t i = 0; i < weights.n_rows; i++)
    for (size_t j = 0; j < weights.n_cols; j++)
      BOOST_REQUIRE_SMALL(weights.at(i, j) - orthogonalWeights.at(i, j), 1e-3);
}

// Test the ZeroInitialization class. If you think about it, it's kind of
// ridiculous to test the zero init rule. But at least we make sure it
// builds without any problems.
BOOST_AUTO_TEST_CASE(ZeroInitTest)
{
  arma::mat weights;
  ZeroInitialization zeroInit;
  zeroInit.Initialize(weights, 100, 100);

  bool b = arma::all(arma::vectorise(weights) == 0);
  BOOST_REQUIRE_EQUAL(b, 1);
}

// Test the KathirvalavakumarSubavathiInitialization class.
BOOST_AUTO_TEST_CASE(KathirvalavakumarSubavathiInitTest)
{
  arma::mat data = arma::randu<arma::mat>(100, 1);

  arma::mat weights;
  KathirvalavakumarSubavathiInitialization kathirvalavakumarSubavathiInit(
      data, 1.5);
  kathirvalavakumarSubavathiInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

// Test the NguyenWidrowInitialization class.
BOOST_AUTO_TEST_CASE(NguyenWidrowInitTest)
{
  arma::mat weights;
  NguyenWidrowInitialization nguyenWidrowInit;
  nguyenWidrowInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

// Test the OivsInitialization class.
BOOST_AUTO_TEST_CASE(OivsInitTest)
{
  arma::mat weights;
  OivsInitialization<> oivsInit;
  oivsInit.Initialize(weights, 100, 100);

  BOOST_REQUIRE_EQUAL(1, 1);
}

// Test the GaussianInitialization class.
BOOST_AUTO_TEST_CASE(GaussianInitTest)
{
  const size_t row = 7;
  const size_t col = 7;
  const size_t slice = 2;

  double mean = 1;
  double mean3d = 1;
  double var = 1;
  double var3d = 1;

  arma::mat weights;
  arma::cube weights3d;

  GaussianInitialization t(0, 0.2);

  // It isn't guaranteed that the method will converge in the specified number
  // of iterations using random weights. If this works 1 of 5 times, I'm fine
  // with that.
  size_t counter = 0;
  for(size_t trial = 0; trial < 5; trial++)
  {
    for(size_t i = 0; i < 10; i++)
    {
      t.Initialize(weights, row, col);
      t.Initialize(weights3d, row, col, slice);

      // Calaculate mean and variance over the dense matrix.
      mean += arma::accu(weights) / weights.n_elem;
      var += arma::accu(pow((weights.t() - mean), 2)) / weights.n_elem - 1;

      // Calaculate mean and variance over the 3rd order tensor.
      mean3d += arma::accu(weights3d.slice(0)) / weights3d.slice(0).n_elem;
      var3d += arma::accu(pow((weights3d.slice(0) - mean), 2)) /
          weights3d.slice(0).n_elem - 1;
    }

    mean /= 10;
    var /= 10;
    mean3d /= 10;
    var3d /= 10;

    if ((mean > 0 && mean < 0.4) && (var > 0 && var < 0.6) &&
        (mean3d > 0 && mean3d < 0.4) && (var3d > 0 && var3d < 0.6))
    {
      counter++;
      break;
    }
  }

  BOOST_REQUIRE(counter >= 1);
}

BOOST_AUTO_TEST_SUITE_END();
