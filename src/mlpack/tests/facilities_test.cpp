/**
 * @file facilities_test.cpp
 * @author Khizir Siddiqui
 *
 * Test file for facilities in metrics.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/data/load.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::cv;

BOOST_AUTO_TEST_SUITE(FacilitiesTest);


/**
 * The unequal sizes for data and labels show throw an error.
 */
BOOST_AUTO_TEST_CASE(AssertSizesTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("iris_train.csv", dataset);
  // Load the labels.
  arma::Row<size_t> labels;
  data::Load("iris_test_labels.csv", labels);

  BOOST_REQUIRE_THROW(
    AssertSizes(dataset, labels, "test"), std::invalid_argument);
}


/**
 * Pairwise distances.
 */
BOOST_AUTO_TEST_CASE(PairwiseDistanceTest)
{
  arma::mat X;
  X = { {0, 1, 1, 0, 0},
        {0, 1, 2, 0, 0},
        {1, 1, 3, 2, 0} };
  metric::EuclideanDistance metric;
  arma::mat dist = PairwiseDistances(X, metric);
  BOOST_REQUIRE_EQUAL(dist(0, 0), 0);
  BOOST_REQUIRE_CLOSE(dist(1, 0), 1.41421, 1e-3);
  BOOST_REQUIRE_EQUAL(dist(2, 0), 3);
}

BOOST_AUTO_TEST_SUITE_END();
