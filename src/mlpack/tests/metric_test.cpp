/**
 * @file tests/metric_test.cpp
 *
 * Unit tests for the various metrics.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/test/unit_test.hpp>
#include <mlpack/core/metrics/iou_metric.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(LMetricTest);

/**
 * Simple test for L-1 metric.
 */
BOOST_AUTO_TEST_CASE(L1MetricTest)
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::Col<size_t> b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  ManhattanDistance lMetric;

  BOOST_REQUIRE_CLOSE((double) arma::accu(arma::abs(a1 - b1)),
                      lMetric.Evaluate(a1, b1), 1e-5);

  BOOST_REQUIRE_CLOSE((double) arma::accu(arma::abs(a2 - b2)),
                      lMetric.Evaluate(a2, b2), 1e-5);
}

/**
 * Simple test for L-2 metric.
 */
BOOST_AUTO_TEST_CASE(L2MetricTest)
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::vec a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::vec b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  EuclideanDistance lMetric;

  BOOST_REQUIRE_CLOSE((double) sqrt(arma::accu(arma::square(a1 - b1))),
                      lMetric.Evaluate(a1, b1), 1e-5);

  BOOST_REQUIRE_CLOSE((double) sqrt(arma::accu(arma::square(a2 - b2))),
                      lMetric.Evaluate(a2, b2), 1e-5);
}

/**
 * Simple test for L-Infinity metric.
 */
BOOST_AUTO_TEST_CASE(LINFMetricTest)
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::Col<size_t> b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  ChebyshevDistance lMetric;

  BOOST_REQUIRE_CLOSE((double) arma::as_scalar(arma::max(arma::abs(a1 - b1))),
                      lMetric.Evaluate(a1, b1), 1e-5);

  BOOST_REQUIRE_CLOSE((double) arma::as_scalar(arma::max(arma::abs(a2 - b2))),
                      lMetric.Evaluate(a2, b2), 1e-5);
}

/**
 * Simple test for IoU metric.
 */
BOOST_AUTO_TEST_CASE(IoUMetricTest)
{
  arma::vec bbox1(4), bbox2(4);
  bbox1 << 1 << 2 << 100 << 200;
  bbox2 << 1 << 2 << 100 << 200;
  // IoU of same bounding boxes equals 1.0.
  BOOST_REQUIRE_CLOSE(1.0, IoU<>::Evaluate(bbox1, bbox2), 1e-4);

  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, x1, y1}.
  bbox1 << 39 << 63 << 203 << 112;
  bbox2 << 54 << 66 << 198 << 114;
  // Value calculated using Python interpreter.
  BOOST_REQUIRE_CLOSE(IoU<true>::Evaluate(bbox1, bbox2), 0.7980093, 1e-4);

  bbox1 << 31 << 69 << 201 << 125;
  bbox2 << 18 << 63 << 235 << 135;
  // Value calculated using Python interpreter.
  BOOST_REQUIRE_CLOSE(IoU<true>::Evaluate(bbox1, bbox2), 0.612479577, 1e-4);

  // Use hieght - width representation of bounding boxes.
  // Bounding boxes represent {x0, y0, h, w}.
  bbox1 << 49 << 75 << 154 << 50;
  bbox2 << 42 << 78 << 144 << 48;
  // Value calculated using Python interpreter.
  BOOST_REQUIRE_CLOSE(IoU<>::Evaluate(bbox1, bbox2), 0.7898879, 1e-4);

  bbox1 << 35 << 51 << 161 << 59;
  bbox2 << 36 << 60 << 144 << 48;
  // Value calculated using Python interpreter.
  BOOST_REQUIRE_CLOSE(IoU<>::Evaluate(bbox1, bbox2), 0.7309670, 1e-4);
}

BOOST_AUTO_TEST_SUITE_END();
