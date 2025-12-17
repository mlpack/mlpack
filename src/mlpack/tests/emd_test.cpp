/**
 * @file tests/emd_test.cpp
 * @author Mohammad Mundiwala
 *
 * Tests for EMD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include "catch.hpp"

#include <mlpack/core/transforms/emd.hpp>
using namespace mlpack;
using namespace arma;

TEST_CASE("EMDToy", "[EMD]")
{
  // Synthetic two tone, zero-trend signal. We expect near-exact
  // reconstruction because the IMFs should capture both sine components and
  // leave a near-zero residue. If this fails, spline envelopes or sifting
  // convergence likely need improvement.
  arma::vec t = arma::linspace<arma::vec>(0.0, 1.0, 1000);
  arma::vec x = arma::sin(2 * arma::datum::pi * 5.0 * t)
              + 0.5 * arma::sin(2 * arma::datum::pi * 20.0 * t);

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  // Check exact reconstruction (IMFs + residue = original signal).
  arma::vec recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Toy two-tone reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-6);
}

TEST_CASE("EMDDiagnosticToy", "[EMD]")
{
  // Mixed frequencies with a  slow linear trend. Small reconstruction
  // error (1e-3) to tolerate spline interpolation/finite sifting. Failure
  // liekly means stopping criterion issue.
  arma::vec t = arma::linspace<arma::vec>(0, 1, 2000);
  arma::vec x = arma::sin(2*datum::pi*3*t) +
                0.3*arma::sin(2*datum::pi*12*t) +
                0.1*t;

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  for (uword k = 0; k < imfs.n_cols; ++k)
    UNSCOPED_INFO("IMF[" << k << "] L2 = " << arma::norm(imfs.col(k)));

  // Reconstruction accuracy on mixed tones with trend
  arma::vec recon = r;
  for (uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  double relErr = arma::norm(recon - x) / arma::norm(x);
  UNSCOPED_INFO("Reconstruction error: " << relErr);

  REQUIRE(relErr < 1e-3);
}

TEST_CASE("EMDDiagnostic", "[EMD]")
{
  // Realistic nonstationary toy signal from CSV. Ensures I/O path function and
  // reconstruction stays accurate within `relErr`. Failures may mean issues with
  // dataset extrema, spline envelope instability due to closely spaced extrema
  vec x;
  REQUIRE(data::Load("nonstationary_signal_toy.csv", x, false));

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  for (uword k = 0; k < imfs.n_cols; ++k)
    UNSCOPED_INFO("IMF[" << k << "] L2 = " << arma::norm(imfs.col(k)));

  // Reconstruction accuracy on csv toy 
  arma::vec recon = r;
  for (uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Reconstruction error: " << relErr);

  REQUIRE(relErr < 1e-3);
}

TEST_CASE("EMDMonotoneNoImf", "[EMD]")
{
  // Monotone input should yield 0 IMFs and residue equals input. This guards
  // the extrema endpoints logic, failure would mean there are
  // IMFs when fewer than two extrema exist.
  arma::vec x = arma::linspace<arma::vec>(0.0, 5.0, 500);

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols == 0);
  REQUIRE(arma::norm(r - x, 2) / arma::norm(x, 2) < 1e-12);
}

TEMPLATE_TEST_CASE("EMDTemplateReconstruction", "[EMD]", float, double)
{
  // Validate reconstruction accuracy for both float and double paths
  // with the templated spline envelope wrapper.
  using eT = TestType;
  arma::Col<eT> t = arma::linspace<arma::Col<eT>>(eT(0), eT(1), 800);
  arma::Col<eT> x = arma::sin(eT(2) * eT(arma::datum::pi) * eT(4) * t)
                  + eT(0.25) * arma::sin(eT(2) * eT(arma::datum::pi) * eT(14) * t);

  arma::Mat<eT> imfs;
  arma::Col<eT> r;
  emd::EMD(x, imfs, r);

  // Extract at least one IMF and achieve accurate reconstruction.
  REQUIRE(imfs.n_cols >= 1);

  arma::Col<eT> recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Template reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-3);
}
