/**
 * @file boostmetric_test.cpp
 * @author Manish Kumar
 *
 * Unit tests for BoostMetric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/boostmetric/boostmetric.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::boostmetric;


BOOST_AUTO_TEST_SUITE(BoostMetricTest);

double BoostMetricKnnAccuracy(const arma::mat& dataset,
                   const arma::Row<size_t>& labels,
                   const size_t k)
{
  arma::Row<size_t> uniqueLabels = arma::unique(labels);

  arma::Mat<size_t> neighbors;
  arma::mat distances;

  neighbor::KNN knn;

  knn.Train(dataset);
  knn.Search(k, neighbors, distances);

  // Keep count.
  size_t count = 0.0;

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    arma::vec Map;
    Map.zeros(uniqueLabels.n_cols);

    for (size_t j = 0; j < k; j++)
      Map(labels(neighbors(j, i))) +=
          1 / std::pow(distances(j, i) + 1, 2);

    size_t index = arma::conv_to<size_t>::from(arma::find(Map
        == arma::max(Map)));

    // Increase count if labels match.
    if (index == labels(i))
        count++;
  }

  // return accuracy.
  return ((double) count / dataset.n_cols) * 100;
}

// Check that final accuracy is greater than initial accuracy on
// simple dataset.
BOOST_AUTO_TEST_CASE(BoostMetricAccuracyTest)
{
  // Useful but simple dataset with six points and two classes.
  arma::mat dataset        = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  double initAccuracy = BoostMetricKnnAccuracy(dataset, labels, 3);

  BoostMetric<> bm(dataset, labels, 1);

  arma::mat outputMatrix;
  bm.LearnDistance(outputMatrix);

  double finalAccuracy =
      BoostMetricKnnAccuracy(outputMatrix * dataset, labels, 3);

  // finalObj must be less than initObj.
  BOOST_REQUIRE_LT(initAccuracy, finalAccuracy);

  // Since this is a very simple dataset final accuracy should be around 100%.
  BOOST_REQUIRE_CLOSE(finalAccuracy, 100.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
