/**
 * @file fastmks_test.cpp
 * @author Ryan Curtin
 *
 * Ensure that fast max-kernel search is correct.
 *
 * This file is part of MLPACK 1.0.7.
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
#include <mlpack/methods/fastmks/fastmks.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::fastmks;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(FastMKSTest);

/**
 * Compare single-tree and naive.
 */
BOOST_AUTO_TEST_CASE(SingleTreeVsNaive)
{
  // First create a random dataset.
  arma::mat data;
  data.randn(5, 1000);
  LinearKernel lk;

  // Now run FastMKS naively.
  FastMKS<LinearKernel> naive(data, lk, false, true);

  arma::Mat<size_t> naiveIndices;
  arma::mat naiveProducts;
  naive.Search(10, naiveIndices, naiveProducts);

  // Now run it in single-tree mode.
  FastMKS<LinearKernel> single(data, lk, true);

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

/**
 * Compare dual-tree and naive.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsNaive)
{
  // First create a random dataset.
  arma::mat data;
  data.randn(10, 5000);
  LinearKernel lk;

  // Now run FastMKS naively.
  FastMKS<LinearKernel> naive(data, lk, false, true);

  arma::Mat<size_t> naiveIndices;
  arma::mat naiveProducts;
  naive.Search(10, naiveIndices, naiveProducts);

  // Now run it in dual-tree mode.
  FastMKS<LinearKernel> tree(data, lk);

  arma::Mat<size_t> treeIndices;
  arma::mat treeProducts;
  tree.Search(10, treeIndices, treeProducts);

  for (size_t q = 0; q < treeIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < treeIndices.n_rows; ++r)
    {
      BOOST_REQUIRE_EQUAL(treeIndices(r, q), naiveIndices(r, q));
      BOOST_REQUIRE_CLOSE(treeProducts(r, q), naiveProducts(r, q), 1e-5);
    }
  }
}

/**
 * Compare dual-tree and single-tree on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(DualTreeVsSingleTree)
{
  // First create a random dataset.
  arma::mat data;
  data.randu(20, 15000);
  PolynomialKernel pk(5.0, 2.5);

  FastMKS<PolynomialKernel> single(data, pk, true);

  arma::Mat<size_t> singleIndices;
  arma::mat singleProducts;
  single.Search(10, singleIndices, singleProducts);

  // Now run it in dual-tree mode.
  FastMKS<PolynomialKernel> tree(data, pk);

  arma::Mat<size_t> treeIndices;
  arma::mat treeProducts;
  tree.Search(10, treeIndices, treeProducts);

  for (size_t q = 0; q < treeIndices.n_cols; ++q)
  {
    for (size_t r = 0; r < treeIndices.n_rows; ++r)
    {
      BOOST_REQUIRE_EQUAL(treeIndices(r, q), singleIndices(r, q));
      BOOST_REQUIRE_CLOSE(treeProducts(r, q), singleProducts(r, q), 1e-5);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
