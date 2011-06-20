/***
 * @file kernel_test.cc
 * @author Ryan Curtin
 *
 * Tests for the various kernel classes.
 */

#include "lmetric.h"
#include "mahalanobis_distance.h"

#define BOOST_TEST_MODULE Kernel Test
#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::kernel;

/***
 * Basic test of the Manhattan distance.
 */
BOOST_AUTO_TEST_CASE(manhattan_distance) {
  // A couple quick tests.
  arma::vec a = "1.0 3.0 4.0";
  arma::vec b = "3.0 3.0 5.0";

  BOOST_REQUIRE_CLOSE(ManhattanDistance::Evaluate(a, b), 3, 1e-5);
  BOOST_REQUIRE_CLOSE(ManhattanDistance::Evaluate(b, a), 3, 1e-5);

  // Check also for when the root is taken (should be the same).
  BOOST_REQUIRE_CLOSE((LMetric<1, true>::Evaluate(a, b)), 3, 1e-5);
  BOOST_REQUIRE_CLOSE((LMetric<1, true>::Evaluate(b, a)), 3, 1e-5);
}

/***
 * Basic test of squared Euclidean distance.
 */
BOOST_AUTO_TEST_CASE(squared_euclidean_distance) {
  // Sample 2-dimensional vectors.
  arma::vec a = "1.0  2.0";
  arma::vec b = "0.0 -2.0";

  BOOST_REQUIRE_CLOSE(SquaredEuclideanDistance::Evaluate(a, b), 17.0, 1e-5);
  BOOST_REQUIRE_CLOSE(SquaredEuclideanDistance::Evaluate(b, a), 17.0, 1e-5);
}

/***
 * Basic test of Euclidean distance.
 */
BOOST_AUTO_TEST_CASE(euclidean_distance) {
  arma::vec a = "1.0 3.0 5.0 7.0";
  arma::vec b = "4.0 0.0 2.0 0.0";

  BOOST_REQUIRE_CLOSE(EuclideanDistance::Evaluate(a, b), sqrt(76.0), 1e-5);
  BOOST_REQUIRE_CLOSE(EuclideanDistance::Evaluate(b, a), sqrt(76.0), 1e-5);
}

/***
 * Arbitrary test case for coverage.
 */
BOOST_AUTO_TEST_CASE(arbitrary_case) {
  arma::vec a = "3.0 5.0 6.0 7.0";
  arma::vec b = "1.0 2.0 1.0 0.0";

  BOOST_REQUIRE_CLOSE(LMetric<3>::Evaluate(a, b), 503.0, 1e-5);
  BOOST_REQUIRE_CLOSE(LMetric<3>::Evaluate(b, a), 503.0, 1e-5);

  BOOST_REQUIRE_CLOSE((LMetric<3, true>::Evaluate(a, b)), 7.95284762, 1e-5);
  BOOST_REQUIRE_CLOSE((LMetric<3, true>::Evaluate(b, a)), 7.95284762, 1e-5);
}

/***
 * Make sure two vectors of all zeros return zero distance, for a few different
 * powers.
 */
BOOST_AUTO_TEST_CASE(lmetric_zeros) {
  arma::vec a(250);
  a.fill(0.0);

  // We cannot use a loop because compilers seem to be unable to unroll the loop
  // and realize the variable actually is knowable at compile-time.
  BOOST_REQUIRE((LMetric<1, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<1, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<2, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<2, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<3, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<3, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<4, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<4, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<5, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<5, true>::Evaluate(a, a)) == 0);
}

/***
 * Simple test of Mahalanobis distance with unset covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_unset_covariance) {
  MahalanobisDistance<false> md;
  arma::vec a = "1.0 2.0 2.0 3.0";
  arma::vec b = "0.0 0.0 1.0 3.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 6.0, 1e-5);
}

/***
 * Simple test of Mahalanobis distance with unset covariance matrix and
 * t_take_root set to true.
 */
BOOST_AUTO_TEST_CASE(md_root_unset_covariance) {
  MahalanobisDistance<true> md;
  arma::vec a = "1.0 2.0 2.5 5.0";
  arma::vec b = "0.0 2.0 0.5 8.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), sqrt(14.0), 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), sqrt(14.0), 1e-5);
}

/***
 * Simple test with diagonal covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_diagonal_covariance) {
  arma::mat cov = arma::eye<arma::mat>(5, 5);
  cov(0, 0) = 2.0;
  cov(1, 1) = 0.5;
  cov(2, 2) = 3.0;
  cov(3, 3) = 1.0;
  cov(4, 4) = 1.5;
  MahalanobisDistance<false> md(cov);

  arma::vec a = "1.0 2.0 2.0 4.0 5.0";
  arma::vec b = "2.0 3.0 1.0 1.0 0.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 52.0, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 52.0, 1e-5);
}

/***
 * More specific case with more difficult covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_full_covariance) {
  arma::mat cov = "1.0 2.0 3.0 4.0;"
                  "0.5 0.6 0.7 0.1;"
                  "3.4 4.3 5.0 6.1;"
                  "1.0 2.0 4.0 1.0;";
  MahalanobisDistance<false> md(cov);

  arma::vec a = "1.0 2.0 2.0 4.0";
  arma::vec b = "2.0 3.0 1.0 1.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 15.7, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 15.7, 1e-5);
}
