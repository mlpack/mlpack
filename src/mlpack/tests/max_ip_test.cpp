/**
 * @file max_ip_test.cpp
 * @author Ryan Curtin
 *
 * Ensure that the maximum inner product search is successful.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/maxip/max_ip.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::maxip;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(MaxIPTest);

/**
 * Compare single-tree and naive.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  // First create a random dataset.
  arma::mat data;
  data.randn(5, 5000);

  // Now run MaxIP naively.
  MaxIP<LinearKernel> naive(data, false, true);

  arma::Mat<size_t> naiveIndices;
  arma::mat naiveProducts;
  naive.Search(10, naiveIndices, naiveProducts);

  // Now run it in single-tree mode.
  MaxIP<LinearKernel> single(data, true);

  arma::Mat<size_t> singleIndices;
  arma::mat singleProducts;
  single.Search(10, singleIndices, singleProducts);

  // Compare the results.
  for (size_t q = 0; q < singleIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < singleIndices.n_rows; ++r)
    {
      BOOST_REQUIRE_EQUAL(singleIndices(r, q), naiveIndices(r, q));
      BOOST_REQUIRE_CLOSE(singleProducts(r, q), naiveProducts(r, q), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
