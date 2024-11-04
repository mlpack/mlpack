/**
 * @file tests/local_coordinate_coding_test.cpp
 * @author Nishant Mehta
 *
 * Test for Local Coordinate Coding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/methods/local_coordinate_coding.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace arma;
using namespace mlpack;

template<typename MatType, typename VecType>
void VerifyCorrectness(const MatType& beta,
                       const VecType& errCorr,
                       const double lambda)
{
  const double tol = 0.1;
  size_t nDims = beta.n_elem;
  for (size_t j = 0; j < nDims; ++j)
  {
    if (beta(j) == 0)
    {
      // make sure that errCorr(j) <= lambda
      REQUIRE(std::max(fabs(errCorr(j)) - lambda, 0.0) ==
          Approx(0.0).margin(tol));
    }
    else if (beta(j) < 0)
    {
      // make sure that errCorr(j) == lambda
      REQUIRE(errCorr(j) - lambda == Approx(0.0).margin(tol));
    }
    else
    { // beta(j) > 0
      // make sure that errCorr(j) == -lambda
      REQUIRE(errCorr(j) + lambda == Approx(0.0).margin(tol));
    }
  }
}

TEMPLATE_TEST_CASE("LocalCoordinateCodingTestCodingStep",
    "[LocalCoordinateCodingTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;
  using VecType = arma::Col<typename MatType::elem_type>;

  double lambda1 = 0.1;
  uword nAtoms = 10;

  mat inX; // The .arm file is saved as an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // normalize each point since these are images
  for (uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  MatType Z;
  LocalCoordinateCoding<MatType> lcc(X, nAtoms, lambda1, 10);
  lcc.Encode(X, Z);

  MatType D = lcc.Dictionary();

  for (uword i = 0; i < nPoints; ++i)
  {
    VecType sqDists(nAtoms);
    for (uword j = 0; j < nAtoms; ++j)
    {
      sqDists[j] = arma::norm(D.col(j) - X.col(i));
    }
    MatType Dprime = D * diagmat(1.0 / sqDists);
    MatType zPrime = Z.unsafe_col(i) % sqDists;

    VecType errCorr = trans(Dprime) * (Dprime * zPrime - X.unsafe_col(i));
    VerifyCorrectness(zPrime, errCorr, 0.5 * lambda1);
  }
}

TEMPLATE_TEST_CASE("LocalCoordinateCodingTestDictionaryStep",
    "[LocalCoordinateCodingTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  const double tol = 0.1;

  double lambda = 0.1;
  uword nAtoms = 10;

  mat inX; // File is saved as an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // normalize each point since these are images
  for (uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  MatType Z;
  LocalCoordinateCoding<MatType> lcc(X, nAtoms, lambda, 10);
  lcc.Encode(X, Z);
  uvec adjacencies = find(Z);
  lcc.OptimizeDictionary(X, Z, adjacencies);

  MatType D = lcc.Dictionary();

  MatType grad = zeros<MatType>(D.n_rows, D.n_cols);
  for (uword i = 0; i < nPoints; ++i)
  {
    grad += (D - repmat(X.unsafe_col(i), 1, nAtoms)) *
        diagmat(abs(Z.unsafe_col(i)));
  }
  grad = lambda * grad + (D * Z - X) * trans(Z);

  REQUIRE(norm(grad, "fro") == Approx(0.0).margin(tol));
}

TEMPLATE_TEST_CASE("LocalCoordinateCodingSerializationTest",
    "[LocalCoordinateCodingTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  MatType X = randu<MatType>(100, 100);
  size_t nAtoms = 10;

  LocalCoordinateCoding<MatType> lcc(nAtoms, 0.05,
      2 /* don't care about quality */);
  lcc.Train(X);

  MatType Y = randu<MatType>(100, 200);
  MatType codes;
  lcc.Encode(Y, codes);

  LocalCoordinateCoding<MatType> lccXml(50, 0.1), lccJson(12, 0.0),
      lccBinary(0, 0.0);
  SerializeObjectAll(lcc, lccXml, lccJson, lccBinary);

  CheckMatrices(lcc.Dictionary(), lccXml.Dictionary(), lccJson.Dictionary(),
      lccBinary.Dictionary());

  MatType xmlCodes, jsonCodes, binaryCodes;
  lccXml.Encode(Y, xmlCodes);
  lccJson.Encode(Y, jsonCodes);
  lccBinary.Encode(Y, binaryCodes);

  CheckMatrices(codes, xmlCodes, jsonCodes, binaryCodes);

  // Check the parameters, too.

  REQUIRE(lcc.Atoms() == lccXml.Atoms());
  REQUIRE(lcc.Atoms() == lccJson.Atoms());
  REQUIRE(lcc.Atoms() == lccBinary.Atoms());

  REQUIRE(lcc.Tolerance() == Approx(lccXml.Tolerance()).epsilon(1e-7));
  REQUIRE(lcc.Tolerance() == Approx(lccJson.Tolerance()).epsilon(1e-7));
  REQUIRE(lcc.Tolerance() == Approx(lccBinary.Tolerance()).epsilon(1e-7));

  REQUIRE(lcc.Lambda() == Approx(lccXml.Lambda()).epsilon(1e-7));
  REQUIRE(lcc.Lambda() == Approx(lccJson.Lambda()).epsilon(1e-7));
  REQUIRE(lcc.Lambda() == Approx(lccBinary.Lambda()).epsilon(1e-7));

  REQUIRE(lcc.MaxIterations() == lccXml.MaxIterations());
  REQUIRE(lcc.MaxIterations() == lccJson.MaxIterations());
  REQUIRE(lcc.MaxIterations() == lccBinary.MaxIterations());
}

/**
 * Test that LocalCoordinateCoding::Train() returns finite final objective
 * value.
 */
TEMPLATE_TEST_CASE("LocalCoordinateCodingTrainReturnObjective",
    "[LocalCoordinateCodingTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  double lambda1 = 0.1;
  uword nAtoms = 10;

  mat inX; // File is saved as arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  LocalCoordinateCoding<MatType> lcc(nAtoms, lambda1, 10);
  double objVal = lcc.Train(X);

  REQUIRE(std::isfinite(objVal) == true);
}
