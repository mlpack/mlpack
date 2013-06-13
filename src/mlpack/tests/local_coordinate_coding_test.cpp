/**
 * @file local_coordinate_coding_test.cpp
 *
 * Test for Local Coordinate Coding
 *
 * This file is part of MLPACK 1.0.6.
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
#include <armadillo>
#include <mlpack/methods/local_coordinate_coding/lcc.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::lcc;

BOOST_AUTO_TEST_SUITE(LocalCoordinateCodingTest);

void VerifyCorrectness(vec beta, vec errCorr, double lambda)
{
  const double tol = 1e-12;
  size_t nDims = beta.n_elem;
  for(size_t j = 0; j < nDims; j++)
  {
    if (beta(j) == 0)
    {
      // make sure that errCorr(j) <= lambda
      BOOST_REQUIRE_SMALL(std::max(fabs(errCorr(j)) - lambda, 0.0), tol);
    }
    else if (beta(j) < 0)
    {
      // make sure that errCorr(j) == lambda
      BOOST_REQUIRE_SMALL(errCorr(j) - lambda, tol);
    }
    else
    { // beta(j) > 0
      // make sure that errCorr(j) == -lambda
      BOOST_REQUIRE_SMALL(errCorr(j) + lambda, tol);
    }
  }
}


BOOST_AUTO_TEST_CASE(LocalCoordinateCodingTestCodingStep)
{
  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // normalize each point since these are images
  for (uword i = 0; i < nPoints; i++)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  LocalCoordinateCoding<> lcc(X, nAtoms, lambda1);
  lcc.OptimizeCode();

  mat D = lcc.Dictionary();
  mat Z = lcc.Codes();

  for(uword i = 0; i < nPoints; i++) {
    vec sq_dists = vec(nAtoms);
    for(uword j = 0; j < nAtoms; j++) {
      vec diff = D.unsafe_col(j) - X.unsafe_col(i);
      sq_dists[j] = dot(diff, diff);
    }
    mat Dprime = D * diagmat(1.0 / sq_dists);
    mat zPrime = Z.unsafe_col(i) % sq_dists;

    vec errCorr = trans(Dprime) * (Dprime * zPrime - X.unsafe_col(i));
    VerifyCorrectness(zPrime, errCorr, 0.5 * lambda1);
  }
}

BOOST_AUTO_TEST_CASE(LocalCoordinateCodingTestDictionaryStep)
{
  const double tol = 1e-12;

  double lambda = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // normalize each point since these are images
  for (uword i = 0; i < nPoints; i++)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  LocalCoordinateCoding<> lcc(X, nAtoms, lambda);
  lcc.OptimizeCode();
  mat Z = lcc.Codes();
  uvec adjacencies = find(Z);
  lcc.OptimizeDictionary(adjacencies);

  mat D = lcc.Dictionary();

  mat grad = zeros(D.n_rows, D.n_cols);
  for (uword i = 0; i < nPoints; i++)
  {
    grad += (D - repmat(X.unsafe_col(i), 1, nAtoms)) *
        diagmat(abs(Z.unsafe_col(i)));
  }
  grad = lambda * grad + (D * Z - X) * trans(Z);

  BOOST_REQUIRE_SMALL(norm(grad, "fro"), tol);

}

/*
BOOST_AUTO_TEST_CASE(LocalCoordinateCodingTestWhole)
{

}
*/


BOOST_AUTO_TEST_SUITE_END();
