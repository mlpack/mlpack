/**
 * @file tests/regularized_svd_test.cpp
 * @author Ranjodh Singh
 *
 * Tests for the tSNE method
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/tsne/tsne.hpp>

#include <catch.hpp>

using namespace mlpack;

TEST_CASE("TSNESimpleTest", "[TSNETest]")
{
  TSNE<> tsne;
  arma::mat X = arma::randu<arma::mat>(10, 100);

  arma::mat Y;
  tsne.Embed(X, Y);

  REQUIRE(Y.n_rows == 2);
  REQUIRE(Y.n_cols == 100);
}
