/**
 * @file metric_test.cpp
 *
 * Unit tests for the 'LMetric' class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(LMetricTest);

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

BOOST_AUTO_TEST_CASE(L2MetricTest)
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::Col<size_t> b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  EuclideanDistance lMetric;

  BOOST_REQUIRE_CLOSE((double) sqrt(arma::accu(arma::square(a1 - b1))),
                      lMetric.Evaluate(a1, b1), 1e-5);

  BOOST_REQUIRE_CLOSE((double) sqrt(arma::accu(arma::square(a2 - b2))),
                      lMetric.Evaluate(a2, b2), 1e-5);
}

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

BOOST_AUTO_TEST_SUITE_END();
