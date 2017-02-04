/**
 * @file sparse_coding_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

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

  SparseCoding sc(nAtoms, lambda1, lambda2);
  mat Z;
  DataDependentRandomInitializer::Initialize(X, 25, sc.Dictionary());
  sc.Encode(X, Z);

  mat D = sc.Dictionary();

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

  BOOST_REQUIRE_SMALL(normGradient, tol);
}

BOOST_AUTO_TEST_CASE(SerializationTest)
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
  BOOST_REQUIRE_EQUAL(sc.Atoms(), scXml.Atoms());
  BOOST_REQUIRE_EQUAL(sc.Atoms(), scText.Atoms());
  BOOST_REQUIRE_EQUAL(sc.Atoms(), scBinary.Atoms());

  BOOST_REQUIRE_CLOSE(sc.Lambda1(), scXml.Lambda1(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.Lambda1(), scText.Lambda1(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.Lambda1(), scBinary.Lambda1(), 1e-5);

  BOOST_REQUIRE_CLOSE(sc.Lambda2(), scXml.Lambda2(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.Lambda2(), scText.Lambda2(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.Lambda2(), scBinary.Lambda2(), 1e-5);

  BOOST_REQUIRE_EQUAL(sc.MaxIterations(), scXml.MaxIterations());
  BOOST_REQUIRE_EQUAL(sc.MaxIterations(), scText.MaxIterations());
  BOOST_REQUIRE_EQUAL(sc.MaxIterations(), scBinary.MaxIterations());

  BOOST_REQUIRE_CLOSE(sc.ObjTolerance(), scXml.ObjTolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.ObjTolerance(), scText.ObjTolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.ObjTolerance(), scBinary.ObjTolerance(), 1e-5);

  BOOST_REQUIRE_CLOSE(sc.NewtonTolerance(), scXml.NewtonTolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.NewtonTolerance(), scText.NewtonTolerance(), 1e-5);
  BOOST_REQUIRE_CLOSE(sc.NewtonTolerance(), scBinary.NewtonTolerance(), 1e-5);
}


BOOST_AUTO_TEST_SUITE_END();
