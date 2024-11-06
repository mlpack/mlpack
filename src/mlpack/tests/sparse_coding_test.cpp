/**
 * @file tests/sparse_coding_test.cpp
 *
 * Test for Sparse Coding
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_coding.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization.hpp"

using namespace arma;
using namespace mlpack;

template<typename VecType>
void SCVerifyCorrectness(const VecType& beta,
                         const VecType& errCorr,
                         double lambda)
{
  const double tol = std::is_same_v<typename VecType::elem_type, float> ?
      1e-6 : 1e-12;
  size_t nDims = beta.n_elem;
  for (size_t j = 0; j < nDims; ++j)
  {
    if (beta(j) == 0)
    {
      // Make sure that errCorr(j) <= lambda.
      REQUIRE(std::max(fabs(errCorr(j)) - lambda, 0.0) ==
          Approx(0.0).margin(tol));
    }
    else if (beta(j) < 0)
    {
      // Make sure that errCorr(j) == lambda.
      REQUIRE(errCorr(j) - lambda == Approx(0.0).margin(tol));
    }
    else // beta(j) > 0.
    {
      // Make sure that errCorr(j) == -lambda.
      REQUIRE(errCorr(j) + lambda == Approx(0.0).margin(tol));
    }
  }
}

TEMPLATE_TEST_CASE("SparseCodingTestCodingStepLasso", "[SparseCodingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using VecType = arma::Col<typename MatType::elem_type>;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  arma::mat inX; // The .arm file contains an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  SparseCoding<MatType> sc(nAtoms, lambda1);
  MatType Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  MatType D = sc.Dictionary();

  for (uword i = 0; i < nPoints; ++i)
  {
    VecType errCorr = trans(D) * (D * Z.unsafe_col(i) - X.unsafe_col(i));
    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

TEMPLATE_TEST_CASE("SparseCodingTestCodingStepElasticNet", "[SparseCodingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;
  using VecType = arma::Col<typename MatType::elem_type>;

  double lambda1 = 0.1;
  double lambda2 = 0.2;
  uword nAtoms = 25;

  arma::mat inX; // The .arm file contains an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding<MatType> sc(nAtoms, lambda1, lambda2);
  MatType Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  MatType D = sc.Dictionary();

  for (uword i = 0; i < nPoints; ++i)
  {
    VecType errCorr =
        (trans(D) * D + lambda2 * eye<MatType>(nAtoms, nAtoms)) *
        Z.unsafe_col(i) - trans(D) * X.unsafe_col(i);

    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

TEMPLATE_TEST_CASE("SparseCodingTestDictionaryStep", "[SparseCodingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;

  const double tol = std::is_same_v<typename MatType::elem_type, float> ?
      0.01 : 1e-6;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  arma::mat inX; // The .arm file contains an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding<MatType> sc(nAtoms, lambda1, 0.0, 0, 0.01, tol);
  MatType Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  MatType D = sc.Dictionary();

  uvec adjacencies = find(Z);
  double normGradient = sc.OptimizeDictionary(X, Z, adjacencies);

  REQUIRE(normGradient == Approx(0.0).margin(tol));
}

TEMPLATE_TEST_CASE("SerializationTest", "[SparseCodingTest]", arma::mat,
    arma::fmat)
{
  using MatType = TestType;

  MatType X = randu<MatType>(100, 100);
  size_t nAtoms = 25;

  SparseCoding<MatType> sc(nAtoms, 0.05, 0.1, 10);
  sc.Train(X);

  MatType Y = randu<MatType>(100, 200);
  MatType codes;
  sc.Encode(Y, codes);

  SparseCoding<MatType> scXml(50, 0.01), scJson(nAtoms, 0.05), scBinary(0, 0.0);
  SerializeObjectAll(sc, scXml, scJson, scBinary);

  CheckMatrices(sc.Dictionary(), scXml.Dictionary(), scJson.Dictionary(),
      scBinary.Dictionary());

  MatType xmlCodes, jsonCodes, binaryCodes;
  scXml.Encode(Y, xmlCodes);
  scJson.Encode(Y, jsonCodes);
  scBinary.Encode(Y, binaryCodes);

  CheckMatrices(codes, xmlCodes, jsonCodes, binaryCodes);

  // Check the parameters, too.
  REQUIRE(sc.Atoms() == scXml.Atoms());
  REQUIRE(sc.Atoms() == scJson.Atoms());
  REQUIRE(sc.Atoms() == scBinary.Atoms());

  REQUIRE(sc.Lambda1() == Approx(scXml.Lambda1()).epsilon(1e-7));
  REQUIRE(sc.Lambda1() == Approx(scJson.Lambda1()).epsilon(1e-7));
  REQUIRE(sc.Lambda1() == Approx(scBinary.Lambda1()).epsilon(1e-7));

  REQUIRE(sc.Lambda2() == Approx(scXml.Lambda2()).epsilon(1e-7));
  REQUIRE(sc.Lambda2() == Approx(scJson.Lambda2()).epsilon(1e-7));
  REQUIRE(sc.Lambda2() == Approx(scBinary.Lambda2()).epsilon(1e-7));

  REQUIRE(sc.MaxIterations() == scXml.MaxIterations());
  REQUIRE(sc.MaxIterations() == scJson.MaxIterations());
  REQUIRE(sc.MaxIterations() == scBinary.MaxIterations());

  REQUIRE(sc.ObjTolerance() == Approx(scXml.ObjTolerance()).epsilon(1e-7));
  REQUIRE(sc.ObjTolerance() == Approx(scJson.ObjTolerance()).epsilon(1e-7));
  REQUIRE(sc.ObjTolerance() == Approx(scBinary.ObjTolerance()).epsilon(1e-7));

  REQUIRE(sc.NewtonTolerance() ==
      Approx(scXml.NewtonTolerance()).epsilon(1e-7));
  REQUIRE(sc.NewtonTolerance() ==
      Approx(scJson.NewtonTolerance()).epsilon(1e-7));
  REQUIRE(sc.NewtonTolerance() ==
      Approx(scBinary.NewtonTolerance()).epsilon(1e-7));
}

/**
 * Test that SparseCoding::Train() returns finite final objective value.
 */
TEMPLATE_TEST_CASE("SparseCodingTrainReturnObjective", "[SparseCodingTest]",
    arma::mat, arma::fmat)
{
  using MatType = TestType;

  const double tol = std::is_same_v<typename MatType::elem_type, float> ?
      0.01 : 1e-6;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  arma::mat inX; // The .arm file contains an arma::mat.
  inX.load("mnist_first250_training_4s_and_9s.csv");
  MatType X = arma::conv_to<MatType>::from(inX);
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding<MatType> sc(nAtoms, lambda1, 0.0, 0, 0.01, tol);
  double objVal = sc.Train(X);

  REQUIRE(std::isfinite(objVal) == true);
}
