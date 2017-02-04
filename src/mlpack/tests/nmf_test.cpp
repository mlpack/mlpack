/**
 * @file nmf_test.cpp
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
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/init_rules/random_acol_init.hpp>
#include <mlpack/methods/amf/init_rules/given_init.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_dist.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

BOOST_AUTO_TEST_SUITE(NMFTest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::amf;

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Default case.
 */
BOOST_AUTO_TEST_CASE(NMFDefaultTest)
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  AMF<> nmf;
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  // Make sure reconstruction error is not too high.  5.0% tolerance.
  BOOST_REQUIRE_SMALL(arma::norm(v - wh, "fro") / arma::norm(v, "fro"),
      0.05);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random Acol initialization distance minimization update.
 */
BOOST_AUTO_TEST_CASE(NMFAcolDistTest)
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  const size_t r = 12;

  SimpleResidueTermination srt(1e-7, 10000);
  AMF<SimpleResidueTermination,RandomAcolInitialization<> >
        nmf(srt);
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  BOOST_REQUIRE_SMALL(arma::norm(v - wh, "fro") / arma::norm(v, "fro"),
      0.015);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix. Random initialization divergence minimization update.
 */
BOOST_AUTO_TEST_CASE(NMFRandomDivTest)
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  // Custom tighter tolerance.
  SimpleResidueTermination srt(1e-8, 10000);
  AMF<SimpleResidueTermination,
      RandomInitialization,
      NMFMultiplicativeDivergenceUpdate> nmf(srt);
  nmf.Apply(v, r, w, h);

  mat wh = w * h;

  // Make sure reconstruction error is not too high.  1.5% tolerance.
  BOOST_REQUIRE_SMALL(arma::norm(v - wh, "fro") / arma::norm(v, "fro"),
      0.015);
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix.  This uses the random initialization and alternating least
 * squares update rule.
 */
BOOST_AUTO_TEST_CASE(NMFALSTest)
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

  // Make sure reconstruction error is not too high.  8% tolerance.  It seems
  // like ALS doesn't converge to results that are as good.  It also seems to be
  // particularly sensitive to initial conditions.
  BOOST_REQUIRE_SMALL(arma::norm(v - wh, "fro") / arma::norm(v, "fro"),
      0.08);
}

/**
 * Check the if the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix. Random Acol initialization,
 * distance minimization update.
 */
BOOST_AUTO_TEST_CASE(SparseNMFAcolDistTest)
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
  BOOST_REQUIRE_SMALL(arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro"),
      1e-5);
}

/**
 * Check that the product of the calculated factorization is close to the
 * input matrix, with a sparse input matrix.  This uses the random
 * initialization and alternating least squares update rule.
 */
BOOST_AUTO_TEST_CASE(SparseNMFALSTest)
{
  // We have to ensure that the residues aren't NaNs.  This can happen when a
  // matrix is created with all zeros in a column or row.
  double denseResidue = std::numeric_limits<double>::quiet_NaN();
  double sparseResidue = std::numeric_limits<double>::quiet_NaN();

  mat vp, dvp; // Resulting matrices.

  while (sparseResidue != sparseResidue && denseResidue != denseResidue)
  {
    mlpack::math::RandomSeed(std::time(NULL));
    mat w, h;
    sp_mat v;
    v.sprandu(10, 10, 0.3);
    // Ensure there is at least one nonzero element in every row and column.
    for (size_t i = 0; i < 10; ++i)
      v(i, i) += 1e-5;
    mat dv(v); // Make a dense copy.
    mat dw, dh;
    size_t r = 5;

    SimpleResidueTermination srt(1e-10, 10000);
    AMF<SimpleResidueTermination, RandomInitialization, NMFALSUpdate> nmf(srt);
    const size_t seed = mlpack::math::RandInt(1000000);
    mlpack::math::RandomSeed(seed);
    nmf.Apply(v, r, w, h);
    mlpack::math::RandomSeed(seed);
    nmf.Apply(dv, r, dw, dh);

    // Reconstruct matrices.
    vp = w * h; // In general vp won't be sparse.
    dvp = dw * dh;

    denseResidue = arma::norm(v - vp, "fro");
    sparseResidue = arma::norm(dv - dvp, "fro");
  }

  // Make sure the results are about equal for the W and H matrices.
  BOOST_REQUIRE_SMALL(arma::norm(vp - dvp, "fro") / arma::norm(vp, "fro"),
      1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
