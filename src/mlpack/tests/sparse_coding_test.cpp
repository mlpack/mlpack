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

// Note: We don't use BOOST_REQUIRE_CLOSE in the code below because we need
// to use FPC_WEAK, and it's not at all intuitive how to do that.

#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization_catch.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::sparse_coding;

void SCVerifyCorrectness(vec beta, vec errCorr, double lambda)
{
  const double tol = 1e-12;
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

TEST_CASE("SparseCodingTestCodingStepLasso", "[SparseCodingTest]")
{
  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
  {
    X.col(i) /= norm(X.col(i), 2);
  }

  SparseCoding sc(nAtoms, lambda1);
  mat Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  mat D = sc.Dictionary();

  for (uword i = 0; i < nPoints; ++i)
  {
    vec errCorr = trans(D) * (D * Z.unsafe_col(i) - X.unsafe_col(i));
    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

TEST_CASE("SparseCodingTestCodingStepElasticNet", "[SparseCodingTest]")
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

  SparseCoding sc(nAtoms, lambda1, lambda2);
  mat Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  mat D = sc.Dictionary();

  for (uword i = 0; i < nPoints; ++i)
  {
    vec errCorr =
      (trans(D) * D + lambda2 * eye(nAtoms, nAtoms)) * Z.unsafe_col(i)
      - trans(D) * X.unsafe_col(i);

    SCVerifyCorrectness(Z.unsafe_col(i), errCorr, lambda1);
  }
}

TEST_CASE("SparseCodingTestDictionaryStep", "[SparseCodingTest]")
{
  const double tol = 1e-6;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding sc(nAtoms, lambda1, 0.0, 0, 0.01, tol);
  mat Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  mat D = sc.Dictionary();

  uvec adjacencies = find(Z);
  double normGradient = sc.OptimizeDictionary(X, Z, adjacencies);

  REQUIRE(normGradient == Approx(0.0).margin(tol));
}

TEST_CASE("SerializationTest", "[SparseCodingTest]")
{
  mat X = randu<mat>(100, 100);
  size_t nAtoms = 25;

  SparseCoding sc(nAtoms, 0.05, 0.1);
  sc.Train(X);

  mat Y = randu<mat>(100, 200);
  mat codes;
  sc.Encode(Y, codes);

  SparseCoding scXml(50, 0.01), scText(nAtoms, 0.05), scBinary(0, 0.0);
  SerializeObjectAll(sc, scXml, scText, scBinary);

  CheckMatrices(sc.Dictionary(), scXml.Dictionary(), scText.Dictionary(),
      scBinary.Dictionary());

  mat xmlCodes, textCodes, binaryCodes;
  scXml.Encode(Y, xmlCodes);
  scText.Encode(Y, textCodes);
  scBinary.Encode(Y, binaryCodes);

  CheckMatrices(codes, xmlCodes, textCodes, binaryCodes);

  // Check the parameters, too.
  REQUIRE(sc.Atoms() == scXml.Atoms());
  REQUIRE(sc.Atoms() == scText.Atoms());
  REQUIRE(sc.Atoms() == scBinary.Atoms());

  REQUIRE(sc.Lambda1() == Approx(scXml.Lambda1()).epsilon(1e-7));
  REQUIRE(sc.Lambda1() == Approx(scText.Lambda1()).epsilon(1e-7));
  REQUIRE(sc.Lambda1() == Approx(scBinary.Lambda1()).epsilon(1e-7));

  REQUIRE(sc.Lambda2() == Approx(scXml.Lambda2()).epsilon(1e-7));
  REQUIRE(sc.Lambda2() == Approx(scText.Lambda2()).epsilon(1e-7));
  REQUIRE(sc.Lambda2() == Approx(scBinary.Lambda2()).epsilon(1e-7));

  REQUIRE(sc.MaxIterations() == scXml.MaxIterations());
  REQUIRE(sc.MaxIterations() == scText.MaxIterations());
  REQUIRE(sc.MaxIterations() == scBinary.MaxIterations());

  REQUIRE(sc.ObjTolerance() == Approx(scXml.ObjTolerance()).epsilon(1e-7));
  REQUIRE(sc.ObjTolerance() == Approx(scText.ObjTolerance()).epsilon(1e-7));
  REQUIRE(sc.ObjTolerance() == Approx(scBinary.ObjTolerance()).epsilon(1e-7));

  REQUIRE(sc.NewtonTolerance() ==
      Approx(scXml.NewtonTolerance()).epsilon(1e-7));
  REQUIRE(sc.NewtonTolerance() ==
      Approx(scText.NewtonTolerance()).epsilon(1e-7));
  REQUIRE(sc.NewtonTolerance() ==
      Approx(scBinary.NewtonTolerance()).epsilon(1e-7));
}

/**
 * Test that SparseCoding::Train() returns finite final objective value.
 */
TEST_CASE("SparseCodingTrainReturnObjective", "[SparseCodingTest]")
{
  const double tol = 1e-6;

  double lambda1 = 0.1;
  uword nAtoms = 25;

  mat X;
  X.load("mnist_first250_training_4s_and_9s.arm");
  uword nPoints = X.n_cols;

  // Normalize each point since these are images.
  for (uword i = 0; i < nPoints; ++i)
    X.col(i) /= norm(X.col(i), 2);

  SparseCoding sc(nAtoms, lambda1, 0.0, 0, 0.01, tol);
  double objVal = sc.Train(X);

  REQUIRE(std::isfinite(objVal) == true);
}
