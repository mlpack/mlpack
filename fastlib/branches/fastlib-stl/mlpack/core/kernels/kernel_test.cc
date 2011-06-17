/***
 * @file kernel_test.cc
 * @author Ryan Curtin
 *
 * Tests for the various kernel classes.
 */

#include "lmetric.h"

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

/* Template parameters cannot be doubles, so the InfinityNorm cannot be created
 * using LMetric<>.  I left the tests here though...

/***
 * Basic test of infinity norm.
 *
BOOST_AUTO_TEST_CASE(infinity_norm) {
  arma::vec a = "3.0 4.0 5.0 6.0 7.0 8.0";
  arma::vec b = "1.0 1.0 1.0 2.0 2.0 1.0";

  BOOST_REQUIRE_CLOSE(InfinityNorm::Evaluate(a, b), 7.0, 1e-5);
  BOOST_REQUIRE_CLOSE(InfinityNorm::Evaluate(b, a), 7.0, 1e-5);
}

/***
 * Corner case: when a stupid user asks for the non-rooted infinity norm, we
 * should answer correctly: unless all elements are zero, the result is
 * infinity.
 
BOOST_AUTO_TEST_CASE(nonrooted_infinity_nonzero) {
  arma::vec a = "1.0 5.0";
  arma::vec b = "2.0 3.0";

  BOOST_REQUIRE(
      LMetric<numeric_limits<double>::infinity, false>::Evaluate(a, b) ==
      numeric_limits<double>::infinity);
  BOOST_REQUIRE(
      LMetric<numeric_limits<double>::infinity, false>::Evaluate(b, a) ==
      numeric_limits<double>::infinity);
}

/***
 * Corner case: a stupid user asks for the non-rooted infinity norm and gives an
 * entirely zero vector to evaluate.  The result should be 1.
 
BOOST_AUTO_TEST_CASE(nonrooted_infinity_zero) {
  arma::vec a = "1.0 5.0";

  BOOST_REQUIRE(
      LMetric<numeric_limits<double>::infinity, false>::Evaluate(a, a) == 0.0);
}
*/


