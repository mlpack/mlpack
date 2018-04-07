/**
 * @file kde_test.cpp
 * @author Roberto Hueso (robertohueso96@gmail.com)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(KDETest);

/**
 * Test if simple case is correct.
 */
BOOST_AUTO_TEST_CASE(KDESimpleTest)
{
  // Transposed reference and query sets because it's easier to read.
  arma::mat reference = { {-1.0, -1.0},
                          {-2.0, -1.0},
                          {-3.0, -2.0},
                          { 1.0,  1.0},
                          { 2.0,  1.0},
                          { 3.0,  2.0} };
  arma::mat query = { { 0.0,  0.5},
                      { 0.4, -3.0},
                      { 0.0,  0.0},
                      {-2.1,  1.0} };
  arma::inplace_trans(reference);
  arma::inplace_trans(query);
  arma::vec estimations = arma::vec(query.n_cols, arma::fill::zeros);
  arma::vec estimations_result = {0.07008107430791211955867225924521335400641,
                                  0.0001001563617562331180753723569587521069479,
                                  0.07658867126520703394465527935608406551182,
                                  0.01028120384800740999553525512055784929544};
  KDE<EuclideanDistance,
      arma::mat,
      GaussianKernel,
      KDTree>
    kde = KDE<>(reference, 1e-8, 0.8, 2);
  kde.Evaluate(query, estimations);
  for (size_t i = 0; i < query.n_cols; ++i)
    BOOST_REQUIRE_EQUAL(estimations[i], estimations_result[i]);
}

BOOST_AUTO_TEST_SUITE_END();
