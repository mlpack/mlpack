/**
 * @file tests/distance_test.cpp
 *
 * Unit tests for the various distance metrics.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization.hpp"

using namespace std;
using namespace mlpack;

/**
 * Basic test of the Manhattan distance.
 */
TEST_CASE("ManhattanDistanceTest", "[DistanceTest]")
{
  // A couple quick tests.
  arma::vec a = "1.0 3.0 4.0";
  arma::vec b = "3.0 3.0 5.0";

  REQUIRE(ManhattanDistance::Evaluate(a, b) == Approx(3.0).epsilon(1e-7));
  REQUIRE(ManhattanDistance::Evaluate(b, a) == Approx(3.0).epsilon(1e-7));

  // Check also for when the root is taken (should be the same).
  REQUIRE((LMetric<1, true>::Evaluate(a, b)) == Approx(3.0).epsilon(1e-7));
  REQUIRE((LMetric<1, true>::Evaluate(b, a)) == Approx(3.0).epsilon(1e-7));
}

/**
 * Basic test of squared Euclidean distance.
 */
TEST_CASE("SquaredEuclideanDistanceTest", "[DistanceTest]")
{
  // Sample 2-dimensional vectors.
  arma::vec a = "1.0  2.0";
  arma::vec b = "0.0 -2.0";

  REQUIRE(SquaredEuclideanDistance::Evaluate(a, b) ==
      Approx(17.0).epsilon(1e-7));
  REQUIRE(SquaredEuclideanDistance::Evaluate(b, a) ==
      Approx(17.0).epsilon(1e-7));
}

/**
 * Basic test of Euclidean distance.
 */
TEST_CASE("EuclideanDistanceTest", "[DistanceTest]")
{
  arma::vec a = "1.0 3.0 5.0 7.0";
  arma::vec b = "4.0 0.0 2.0 0.0";

  REQUIRE(EuclideanDistance::Evaluate(a, b) ==
      Approx(sqrt(76.0)).epsilon(1e-7));
  REQUIRE(EuclideanDistance::Evaluate(b, a) ==
      Approx(sqrt(76.0)).epsilon(1e-7));
}

/**
 * Arbitrary test case for coverage.
 */
TEST_CASE("ArbitraryCaseTest", "[DistanceTest]")
{
  arma::vec a = "3.0 5.0 6.0 7.0";
  arma::vec b = "1.0 2.0 1.0 0.0";

  REQUIRE((LMetric<3, false>::Evaluate(a, b)) == Approx(503.0).epsilon(1e-7));
  REQUIRE((LMetric<3, false>::Evaluate(b, a)) == Approx(503.0).epsilon(1e-7));

  REQUIRE((LMetric<3, true>::Evaluate(a, b)) ==
      Approx(7.95284762).epsilon(1e-7));
  REQUIRE((LMetric<3, true>::Evaluate(b, a)) ==
      Approx(7.95284762).epsilon(1e-7));
}

/**
 * Make sure two vectors of all zeros return zero distance, for a few different
 * powers.
 */
TEST_CASE("LMetricZerosTest", "[DistanceTest]")
{
  arma::vec a(250);
  a.fill(0.0);

  // We cannot use a loop because compilers seem to be unable to unroll the loop
  // and realize the variable actually is knowable at compile-time.
  REQUIRE(LMetric<1, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<1, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<2, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<2, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<3, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<3, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<4, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<4, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<5, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<5, true>::Evaluate(a, a) == 0);
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix in
 * constructor.
 */
TEMPLATE_TEST_CASE("MDUnsetCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  MahalanobisDistance<false, arma::Mat<eT>> md;
  md.Q() = arma::eye<arma::Mat<eT>>(4, 4);
  arma::Col<eT> a = "1.0 2.0 2.0 3.0";
  arma::Col<eT> b = "0.0 0.0 1.0 3.0";

  REQUIRE(md.Evaluate(a, b) == Approx(6.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(6.0).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix in
 * constructor and t_take_root set to true.
 */
TEMPLATE_TEST_CASE("MDRootUnsetCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  MahalanobisDistance<true, arma::Mat<eT>> md;
  md.Q() = arma::eye<arma::Mat<eT>>(4, 4);
  arma::Col<eT> a = "1.0 2.0 2.5 5.0";
  arma::Col<eT> b = "0.0 2.0 0.5 8.0";

  REQUIRE(md.Evaluate(a, b) == Approx(sqrt(14.0)).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(sqrt(14.0)).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance setting identity covariance in
 * constructor.
 */
TEMPLATE_TEST_CASE("MDEyeCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  MahalanobisDistance<false, arma::Mat<eT>> md(4);
  arma::Col<eT> a = "1.0 2.0 2.0 3.0";
  arma::Col<eT> b = "0.0 0.0 1.0 3.0";

  REQUIRE(md.Evaluate(a, b) == Approx(6.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(6.0).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance setting identity covariance in
 * constructor and t_take_root set to true.
 */
TEMPLATE_TEST_CASE("MDRootEyeCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  MahalanobisDistance<true, arma::Mat<eT>> md(4);
  arma::Col<eT> a = "1.0 2.0 2.5 5.0";
  arma::Col<eT> b = "0.0 2.0 0.5 8.0";

  REQUIRE(md.Evaluate(a, b) == Approx(sqrt(14.0)).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(sqrt(14.0)).epsilon(1e-7));
}

/**
 * Simple test with diagonal covariance matrix.
 */
TEMPLATE_TEST_CASE("MDDiagonalCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  arma::Mat<eT> q = arma::eye<arma::Mat<eT>>(5, 5);
  q(0, 0) = 2.0;
  q(1, 1) = 0.5;
  q(2, 2) = 3.0;
  q(3, 3) = 1.0;
  q(4, 4) = 1.5;
  MahalanobisDistance<false, arma::Mat<eT>> md(std::move(q));

  arma::Col<eT> a = "1.0 2.0 2.0 4.0 5.0";
  arma::Col<eT> b = "2.0 3.0 1.0 1.0 0.0";

  REQUIRE(md.Evaluate(a, b) == Approx(52.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(52.0).epsilon(1e-7));
}

/**
 * More specific case with more difficult covariance matrix.
 */
TEMPLATE_TEST_CASE("MDFullCovarianceTest", "[DistanceTest]", float, double)
{
  using eT = TestType;

  arma::Mat<eT> q = "1.0 2.0 3.0 4.0;"
                    "0.5 0.6 0.7 0.1;"
                    "3.4 4.3 5.0 6.1;"
                    "1.0 2.0 4.0 1.0;";
  MahalanobisDistance<false, arma::Mat<eT>> md(std::move(q));

  arma::Col<eT> a = "1.0 2.0 2.0 4.0";
  arma::Col<eT> b = "2.0 3.0 1.0 1.0";

  REQUIRE(md.Evaluate(a, b) == Approx(15.7).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(15.7).epsilon(1e-7));
}

/**
 * Simple test for L-1 metric.
 */
TEST_CASE("L1MetricTest", "[DistanceTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 = { 1, 2, 1, 0, 5 };

  arma::Col<size_t> b2(5);
  b2 = { 2, 5, 2, 0, 1 };

  ManhattanDistance lMetric;

  REQUIRE((double) accu(arma::abs(a1 - b1)) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) accu(arma::abs(a2 - b2)) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for L-2 metric.
 */
TEST_CASE("L2MetricTest", "[DistanceTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::vec a2(5);
  a2 = { 1, 2, 1, 0, 5 };

  arma::vec b2(5);
  b2 = { 2, 5, 2, 0, 1 };

  EuclideanDistance lMetric;

  REQUIRE((double) sqrt(accu(square(a1 - b1))) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) sqrt(accu(square(a2 - b2))) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for L-Infinity metric.
 */
TEST_CASE("LINFMetricTest", "[DistanceTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 = { 1, 2, 1, 0, 5 };

  arma::Col<size_t> b2(5);
  b2 = { 2, 5, 2, 0, 1 };

  ChebyshevDistance lMetric;

  REQUIRE((double) arma::as_scalar(arma::max(arma::abs(a1 - b1))) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) arma::as_scalar(arma::max(arma::abs(a2 - b2))) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for IoU distance.
 */
TEST_CASE("IoUDistanceTest", "[DistanceTest]")
{
  arma::vec bbox1(4), bbox2(4);
  bbox1 = { 1, 2, 100, 200 };
  bbox2 = { 1, 2, 100, 200 };
  // IoU of same bounding boxes equals 0.0.
  REQUIRE(0.0 == Approx(IoUDistance<>::Evaluate(bbox1, bbox2)).epsilon(1e-6));

  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, x1, y1}.
  bbox1 = { 39, 63, 203, 112 };
  bbox2 = { 54, 66, 198, 114 };
  // Value calculated using Python interpreter.
  REQUIRE(IoUDistance<true>::Evaluate(bbox1, bbox2) ==
      Approx(1.0 - 0.7980093).epsilon(1e-6));

  bbox1 = { 31, 69, 201, 125 };
  bbox2 = { 18, 63, 235, 135 };
  // Value calculated using Python interpreter.
  REQUIRE(IoUDistance<true>::Evaluate(bbox1, bbox2) ==
      Approx(1.0 - 0.612479577).epsilon(1e-6));

  // Use hieght - width representation of bounding boxes.
  // Bounding boxes represent {x0, y0, h, w}.
  bbox1 = { 49, 75, 154, 50 };
  bbox2 = { 42, 78, 144, 48 };
  // Value calculated using Python interpreter.
  REQUIRE(IoUDistance<>::Evaluate(bbox1, bbox2) ==
      Approx(1.0 - 0.7898879).epsilon(1e-6));

  bbox1 = { 35, 51, 161, 59 };
  bbox2 = { 36, 60, 144, 48 };
  // Value calculated using Python interpreter.
  REQUIRE(IoUDistance<>::Evaluate(bbox1, bbox2) ==
      Approx(1.0 - 0.7309670).epsilon(1e-6));
}

/**
 * Mahalanobis Distance serialization test.
 */
TEST_CASE("MahalanobisDistanceSerializationTest", "[DistanceTest]")
{
  MahalanobisDistance<> d;
  d.Q().randu(50, 50);

  MahalanobisDistance<> xmlD, jsonD, binaryD;

  SerializeObjectAll(d, xmlD, jsonD, binaryD);

  // Check the covariance matrices.
  CheckMatrices(d.Q(), xmlD.Q(), jsonD.Q(), binaryD.Q());
}
