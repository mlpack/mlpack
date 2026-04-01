/**
 * @file tests/emd_test.cpp
 * @author Mohammad Mundiwala
 *
 * Tests for EMD and consequently spline envelope 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include "catch.hpp"

using namespace mlpack;
using namespace arma;

TEST_CASE("EMDSingleTone", "[EMD]")
{
  const arma::uword N = 50000;
  arma::vec t = linspace<arma::vec>(0.0, 1.0, N);
  arma::vec Sin = sin(2.0 * datum::pi * 5.0 * t);
  arma::vec noise = 0.05 * randn<arma::vec>(N);
  arma::vec x = Sin + noise;
  arma::mat imfs;
  arma::vec r;

  // run emd
  // maxImfs=5, maxSiftIter=50, tol=1e-3
  mlpack::EMD(x, imfs, r, 5, 50, 1e-3);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  REQUIRE(imfs.n_cols >= 1);

  const arma::vec imf0 = imfs.col(0);

  // Reconstruction should be almost exact
  arma::vec recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  // Does the reconstruction match the original toy input
  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-3);
}

//edge case test
TEST_CASE("EMDMonotone", "[EMD]")
{
  // Monotone input should yield 0 IMFs and residue equals input
  arma::vec x = linspace<arma::vec>(0.0, 5.0, 500);

  arma::mat imfs;
  arma::vec r;
  mlpack::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols == 0);
  const double relResid = norm(r - x, 2) / norm(x, 2);
  REQUIRE(relResid < 1e-12);
}

//test templates to check for both float and double
TEMPLATE_TEST_CASE("EMDTemplateReconstruction", "[EMD]", float, double)
{
  using eT = TestType;

  arma::Col<eT> t = linspace<arma::Col<eT>>(eT(0), eT(1), 800);
  // x = sin(2*pi*4*t) + 0.25*sin(2*pi*14*t)
  arma::Col<eT> x = sin(eT(2)*datum::pi*eT(4)*t)
                  + eT(0.25) * sin(eT(2)*datum::pi*eT(14)*t);

  arma::Mat<eT> imfs;
  arma::Col<eT> r;
  mlpack::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols >= 1);

  arma::Col<eT> recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = norm(recon - x, 2) / norm(x, 2);
  UNSCOPED_INFO("Template reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-3);
}

TEST_CASE("EEMD", "[EMD]")
{
  arma::arma_rng::set_seed(67);

  const arma::uword N = 5000;
  const double T = 1.0;

  arma::vec t = arma::linspace<arma::vec>(0.0, T, N);
  arma::vec signal =
      arma::sin(2.0 * arma::datum::pi * 50.0 * t) +
      0.6 * arma::sin(2.0 * arma::datum::pi * 12.0 * t) +
      0.3 * arma::sin(2.0 * arma::datum::pi * 3.0 * t);

  arma::mat imfs;
  arma::vec residue;

  mlpack::EEMD(signal, imfs, residue);

  REQUIRE(imfs.n_cols >= 3);

  arma::vec recon = arma::sum(imfs, 1) + residue;
  const double relErr = arma::norm(recon - signal, 2) / arma::norm(signal, 2);
  UNSCOPED_INFO("Reconstruction relErr = " << relErr);
  REQUIRE(relErr < 1e-2);

  // want to check that eemd outputs are reasonable
  const double freqs[3] = {50.0, 12.0, 3.0};
  const double dt = t(1) - t(0);
  const double fs = 1.0 / dt;
  const size_t numToScan = std::min<size_t>(5, imfs.n_cols);
  arma::vec foundPeaks(numToScan);
  for (size_t k = 0; k < numToScan; ++k)
  {
    arma::cx_vec spectrum = arma::fft(imfs.col(k));
    arma::vec mag = arma::abs(spectrum.rows(0, spectrum.n_elem / 2));
    size_t idx = mag.index_max();
    double peakHz = double(idx) * fs / double(spectrum.n_elem);

    foundPeaks(k) = peakHz;
    UNSCOPED_INFO("IMF " << k << " peak = " << peakHz << " Hz");
  }

  // see if first few imfs have the expected peak freqs
  for (size_t j = 0; j < 3; ++j)
  {
    arma::vec err = arma::abs(foundPeaks - freqs[j]);
    double bestErr = err.min();

    UNSCOPED_INFO("Expected " << freqs[j] << " Hz, error = " << bestErr);
    REQUIRE(bestErr < 2.0);
  }
}
