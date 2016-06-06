#include <mlpack/core.hpp>
//#include <mlpack/methods/edge_boxes/structured_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"
BOOST_AUTO_TEST_SUITE(ind2sub_test);

/**
 * This tests handles the case wherein only one class exists in the input
 * labels.  It checks whether the only class supplied was the only class
 * predicted.
 */
BOOST_AUTO_TEST_CASE(ind2sub_test)
{
  arma::mat A = arma::randu(5,5);
  arma::uvec u = arma::ind2sub(arma::size(A), 3);
  u.print();
}
BOOST_AUTO_TEST_SUITE_END();
