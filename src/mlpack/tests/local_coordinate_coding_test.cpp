/**
 * @file local_coordinate_coding_test.cpp
 * @author Nishant Mehta
 *
 * Test for Local Coordinate Coding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.
#include <mlpack/methods/local_coordinate_coding/lcc.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

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

  mat Z;
  LocalCoordinateCoding lcc(X, nAtoms, lambda1);
  lcc.Encode(X, Z);

  mat D = lcc.Dictionary();

  for (uword i = 0; i < nPoints; i++)
  {
    vec sqDists = vec(nAtoms);
    for (uword j = 0; j < nAtoms; j++)
    {
      vec diff = D.unsafe_col(j) - X.unsafe_col(i);
      sqDists[j] = dot(diff, diff);
    }
    mat Dprime = D * diagmat(1.0 / sqDists);
    mat zPrime = Z.unsafe_col(i) % sqDists;

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

  mat Z;
  LocalCoordinateCoding lcc(X, nAtoms, lambda);
  lcc.Encode(X, Z);
  uvec adjacencies = find(Z);
  lcc.OptimizeDictionary(X, Z, adjacencies);

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

BOOST_AUTO_TEST_CASE(SerializationTest)
{
  mat X = randu<mat>(100, 100);
  size_t nAtoms = 25;

  LocalCoordinateCoding lcc(nAtoms, 0.05);
  lcc.Train(X);

  mat Y = randu<mat>(100, 200);
  mat codes;
  lcc.Encode(Y, codes);

  LocalCoordinateCoding lccXml(50, 0.1), lccText(12, 0.0), lccBinary(0, 0.0);
  SerializeObjectAll(lcc, lccXml, lccText, lccBinary);

  CheckMatrices(lcc.Dictionary(), lccXml.Dictionary(), lccText.Dictionary(),
      lccBinary.Dictionary());

  mat xmlCodes, textCodes, binaryCodes;
  lccXml.Encode(Y, xmlCodes);
  lccText.Encode(Y, textCodes);
  lccBinary.Encode(Y, binaryCodes);

  CheckMatrices(codes, xmlCodes, textCodes, binaryCodes);

  // Check the parameters, too.
  BOOST_REQUIRE_EQUAL(lcc.Atoms(), lccXml.Atoms());
  BOOST_REQUIRE_EQUAL(lcc.Atoms(), lccText.Atoms());
  BOOST_REQUIRE_EQUAL(lcc.Atoms(), lccBinary.Atoms());

  BOOST_REQUIRE_CLOSE(lcc.Tolerance(), lccXml.Tolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(lcc.Tolerance(), lccText.Tolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(lcc.Tolerance(), lccBinary.Tolerance(), 1e-5);

  BOOST_REQUIRE_CLOSE(lcc.Lambda(), lccXml.Lambda(), 1e-5);
  BOOST_REQUIRE_CLOSE(lcc.Lambda(), lccText.Lambda(), 1e-5);
  BOOST_REQUIRE_CLOSE(lcc.Lambda(), lccBinary.Lambda(), 1e-5);

  BOOST_REQUIRE_EQUAL(lcc.MaxIterations(), lccXml.MaxIterations());
  BOOST_REQUIRE_EQUAL(lcc.MaxIterations(), lccText.MaxIterations());
  BOOST_REQUIRE_EQUAL(lcc.MaxIterations(), lccBinary.MaxIterations());
}

BOOST_AUTO_TEST_SUITE_END();
