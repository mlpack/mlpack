/**
 * @file sparse_coding_test.cpp
 *
 * Test for Sparse Coding
 * This file is part of MLPACK 1.0.2.
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

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.

#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::sparse_coding;

BOOST_AUTO_TEST_SUITE(SparseCodingTest);

void SCVerifyCorrectness(vec beta, vec errCorr, double lambda)
{
  const double tol = 1e-12;
  size_t nDims = beta.n_elem;
  for(size_t j = 0; j < nDims; j++)
  {
    if (beta(j) == 0)
    {
      // Make sure that errCorr(j) <= lambda.
      BOOST_REQUIRE_SMALL(std::max(fabs(errCorr(j)) - lambda, 0.0), tol);
    }
    else if (beta(j) < 0)
    {
      // Make sure that errCorr(j) == lambda.
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else // beta(j) > 0.
    {
      // Make sure that errCorr(j) == -lambda.
      BOOST_REQUIRE_SMALL(errCorr(j) + lambda, tol);
    }
  }
}

BOOST_AUTO_TEST_CASE(SparseCodingTestCodingStepLasso)
{
  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i) {
    X.col(i) /= norm(X.col(i), 2);
  }

  SparseCoding<> sc(X, nAtoms, lambda1);
  sc.OptimizeCode();

  mat D = sc.Dictionary();
  mat Z = sc.Codes();

  for (uword i = 0; i < nPoints; ++i)
  {
    vec errCorr = trans(D) * (D * Z.unsafe_col(i) - X.unsafe_col(i));
    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

BOOST_AUTO_TEST_CASE(SparseCodingTestCodingStepElasticNet)
{
  double lambda1 = 0.1;
  double lambda2 = 0.2;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding<> sc(X, nAtoms, lambda1, lambda2);
  sc.OptimizeCode();

  mat D = sc.Dictionary();
  mat Z = sc.Codes();

  for(uword i = 0; i < nPoints; ++i)
  {
    vec errCorr =
      (trans(D) * D + lambda2 * eye(nAtoms, nAtoms)) * Z.unsafe_col(i)
      - trans(D) * X.unsafe_col(i);

    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

BOOST_AUTO_TEST_CASE(SparseCodingTestDictionaryStep)
{
  const double tol = 2e-7;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding<> sc(X, nAtoms, lambda1);
  sc.OptimizeCode();

  mat D = sc.Dictionary();
  mat Z = sc.Codes();

  uvec adjacencies = find(Z);
  double normGradient = sc.OptimizeDictionary(adjacencies, 1e-12);

  BOOST_REQUIRE_SMALL(normGradient, tol);
}

/*
BOOST_AUTO_TEST_CASE(SparseCodingTestWhole)
{

}
*/


BOOST_AUTO_TEST_SUITE_END();
