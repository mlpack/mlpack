/**
 * @file tests/nmf_test.cpp
 * @author Mohan Rajendran
 *
 * Test file for NMF class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/nmf.hpp>

#include "catch.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Default case.
 */
TEST_CASE("NMFDefaultTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  AMF<> nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  // Make sure reconstruction error is not too high.  5.0% tolerance.
  REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
      Approx(0.0).margin(0.05));
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Acol initialization distance minimization update.
 */
TEST_CASE("NMFAcolDistTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  const size_t r = 12;

  SimpleResidueTermination srt(1e-7, 10000);
  AMF<SimpleResidueTermination, RandomAcolInitialization<> > nmf(srt);
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
      Approx(0.0).margin(0.15));
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random initialization divergence minimization update.
 */
TEST_CASE("NMFRandomDivTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  const size_t trials = 3;
  bool success = false;

  for (size_t trial = 0; trial < trials; ++trial)
  {
    // Custom tighter tolerance.
    SimpleResidueTermination srt(1e-8, 10000);
    AMF<SimpleResidueTermination,
        RandomAMFInitialization,
        NMFMultiplicativeDivergenceUpdate> nmf(srt);
    nmf.Apply(v, r, w, h);

    mat wh = w * h;

    // Make sure reconstruction error is not too high.  1.5% tolerance.
    if ((arma::norm(v - wh, "fro") / arma::norm(v, "fro")) < 0.015)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix.  This uses the random initialization and alternating least
 * squares update rule.
 */
TEST_CASE("NMFALSTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  SimpleResidueTermination srt(1e-12, 50000);
  AMF<SimpleResidueTermination, RandomAcolInitialization<>, NMFALSUpdate>
        nmf(srt);
  nmf.Apply(v, r, w, h);

  const mat wh = w * h;

  // Make sure reconstruction error is not too high.  9% tolerance.  It seems
  // like ALS doesn't converge to results that are as good.  It also seems to be
  // particularly sensitive to initial conditions.
  REQUIRE(arma::norm(v - wh, "fro") / arma::norm(v, "fro") ==
      Approx(0.0).margin(0.09));
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix. Random Acol initialization,
 * distance minimization update.
 */
TEST_CASE("SparseNMFAcolDistTest", "[NMFTest]")
{
  // We have to ensure that the residues aren't NaNs.  This can happen when a
  // matrix is created with all zeros in a column or row.
  double denseResidue = std::numeric_limits<double>::quiet_NaN();
  double sparseResidue = std::numeric_limits<double>::quiet_NaN();

  mat vp, dvp; // Resulting matrices.

  while (sparseResidue != sparseResidue && denseResidue != denseResidue)
  {
    mat w, h;
    sp_mat v;
    v.sprandu(20, 20, 0.3);
    // Ensure there is at least one nonzero element in every row and column.
    for (size_t i = 0; i < 20; ++i)
      v(i, i) += 1e-5;
    mat dv(v); // Make a dense copy.
    mat dw, dh;
    size_t r = 15;

    SimpleResidueTermination srt(1e-10, 10000);

    // Get an initialization.
    arma::mat iw, ih;
    RandomAcolInitialization<>::Initialize(v, r, iw, ih);
    GivenInitialization g(std::move(iw), std::move(ih));

    // The GivenInitialization will force the same initialization for both
    // Apply() calls.
    AMF<SimpleResidueTermination, GivenInitialization> nmf(srt, g);
    nmf.Apply(v, r, w, h);
    nmf.Apply(dv, r, dw, dh);

    // Reconstruct matrices.
    vp = w * h;
    dvp = dw * dh;

    denseResidue = arma::norm(v - vp, "fro");
    sparseResidue = arma::norm(dv - dvp, "fro");
  }

  // Make sure the results are about equal for the W and H matrices.
  REQUIRE(arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro") ==
      Approx(0.0).margin(1e-5));
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix.  This uses the random
 * initialization and alternating least squares update rule.
 */
TEST_CASE("SparseNMFALSTest", "[NMFTest]")
{
  // We have to ensure that the residues aren't NaNs.  This can happen when a
  // matrix is created with all zeros in a column or row.
  double denseResidue = std::numeric_limits<double>::quiet_NaN();
  double sparseResidue = std::numeric_limits<double>::quiet_NaN();

  mat vp, dvp; // Resulting matrices.

  // We run the test multiple times, since it sometimes fails, in order to get
  // the probability of failure down.
  bool success = false;
  const size_t trials = 8;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    while (sparseResidue != sparseResidue && denseResidue != denseResidue)
    {
      mat w, h;
      sp_mat v;
      v.sprandu(10, 10, 0.3);
      // Ensure there is at least one nonzero element in every row and column.
      for (size_t i = 0; i < 10; ++i)
        v(i, i) += 1e-5;
      mat dv(v); // Make a dense copy.
      mat dw, dh;
      size_t r = 5;

      // Get an initialization.
      arma::mat iw, ih;
      RandomAcolInitialization<>::Initialize(v, r, iw, ih);
      GivenInitialization g(std::move(iw), std::move(ih));

      SimpleResidueTermination srt(1e-10, 10000);
      AMF<SimpleResidueTermination, GivenInitialization, NMFALSUpdate> nmf(srt,
          g);
      nmf.Apply(v, r, w, h);
      nmf.Apply(dv, r, dw, dh);

      // Reconstruct matrices.
      vp = w * h; // In general vp won't be sparse.
      dvp = dw * dh;

      denseResidue = arma::norm(v - vp, "fro");
      sparseResidue = arma::norm(dv - dvp, "fro");
    }

    // Make sure the results are about equal for the W and H matrices.
    const double relDiff = arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro");
    if (relDiff < 1e-5)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Check if all elements in W and H are non-negative.
 * Default Case.
 * Random Acol initilization and distance minimization update.
 */
TEST_CASE("NonNegNMFDefaultTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  const size_t r = 12;

  AMF<> nmf;
  nmf.Apply(v, r, w, h);

  REQUIRE((arma::all(arma::vectorise(w) >= 0)
      && arma::all(arma::vectorise(h) >= 0)));
}

/**
 * Check if all elements in W and H are non-negative.
 * Random initialization divergence minimization update.
 */
TEST_CASE("NonNegNMFRandomDivTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  const size_t r = 12;

  // Custom tighter tolerance.
  SimpleResidueTermination srt(1e-8, 10000);
  AMF<SimpleResidueTermination,
      RandomAMFInitialization,
      NMFMultiplicativeDivergenceUpdate> nmf(srt);
  nmf.Apply(v, r, w, h);

  REQUIRE((arma::all(arma::vectorise(w) >= 0)
      && arma::all(arma::vectorise(h) >= 0)));
}

/**
 * Check if all elements in W and H are non-negative.
 * Random initialization, alternating least squares update.
 */
TEST_CASE("NonNegNMFALSTest", "[NMFTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  const size_t r = 12;

  SimpleResidueTermination srt(1e-12, 50000);
  AMF<SimpleResidueTermination,
      RandomAcolInitialization<>,
      NMFALSUpdate> nmf(srt);
  nmf.Apply(v, r, w, h);

  REQUIRE((arma::all(arma::vectorise(w) >= 0)
      && arma::all(arma::vectorise(h) >= 0)));
}
