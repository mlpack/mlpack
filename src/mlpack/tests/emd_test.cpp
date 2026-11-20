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

TEST_CASE("EEMDOutput", "[EMD]")
{
  const arma::uword N = 3000;

  // signal used in docs (emd.md)
  arma::vec t = arma::linspace<arma::vec>(0.0, 2 * arma::datum::pi, N);
  arma::vec signal =
      arma::sin((20 * t) % (1 + 0.2 * t)) +
      arma::square(t) +
      arma::sin(13 * t);

  arma::mat imfs;
  arma::vec residue;
  mlpack::EEMD(signal, imfs, residue, 300, 0.1, 10, 50, 1e-3);
  // Check reconstruction of signal from imfs + residue
  arma::vec recon = arma::sum(imfs, 1) + residue;
  const double relErr = arma::norm(recon - signal, 2) / arma::norm(signal, 2);
  UNSCOPED_INFO("Reconstruction relErr = " << relErr);
  REQUIRE(relErr < 1e-2);

  // Check that eemd outputs are close to expected freqs.
  // Chirp portion of signal causes frequency to increase linearly over time
  // first IMF frequency defined by (20 + 8 T) / 2pi
  // checks below use the evaluated value at the midpoint of each segment
  // second IMF is due to stationary term defined by 13/2pi ~ 2Hz
  // inspect quarter segments of imf(0)
  const double dt = t(1) - t(0);
  auto ZeroCrossFreq = [&](const arma::vec& seg)
  {
    arma::vec x = seg - arma::mean(seg);

    if (x.n_elem < 2)
      return 0.0;

    const arma::vec left = x.subvec(0, x.n_elem - 2);
    const arma::vec right = x.subvec(1, x.n_elem - 1);

    const arma::uvec crossings =
        ((left >= 0.0) % (right < 0.0)) +
        ((left < 0.0) % (right >= 0.0));

    const size_t zc = arma::accu(crossings);
    const double duration = (x.n_elem - 1) * dt;
    return 0.5 * static_cast<double>(zc) / duration;
  };

  bool foundChirpImf = false;
  bool foundStationaryImf = false;
  const double err = 0.5;

  for (size_t i = 0; i < imfs.n_cols; ++i)
  {
    arma::vec firstQuart  = imfs.col(i).rows(0, N / 4);
    arma::vec secQuart    = imfs.col(i).rows(N / 4, N / 2);
    arma::vec thirdQuart  = imfs.col(i).rows(N / 2, 3 * N / 4);
    arma::vec fourthQuart = imfs.col(i).rows(3 * N / 4, N - 1);

    double avgFreq1 = ZeroCrossFreq(firstQuart);
    double avgFreq2 = ZeroCrossFreq(secQuart);
    double avgFreq3 = ZeroCrossFreq(thirdQuart);
    double avgFreq4 = ZeroCrossFreq(fourthQuart);

    bool isChirpImf =
        (std::abs(avgFreq1 - 4.2) < err) &&
        (std::abs(avgFreq2 - 6.2) < err) &&
        (std::abs(avgFreq3 - 8.2) < err) &&
        (std::abs(avgFreq4 - 10.2) < err);

    if (!foundChirpImf && isChirpImf)
    {
      foundChirpImf = true;

      for (size_t j = i + 1; j < imfs.n_cols; ++j)
      {
        arma::vec stationaryIMF = imfs.col(j).rows(0, N - 1);
        double stationaryFreq = ZeroCrossFreq(stationaryIMF);

        if (std::abs(stationaryFreq - 2.0) < err)
        {
          foundStationaryImf = true;
          break;
        }
      }
      break;
    }
  }

  REQUIRE(foundChirpImf);
  REQUIRE(foundStationaryImf);
}

TEST_CASE("EEMDvsEMD", "[EMD]")
{
  const arma::uword N = 3000;

  // signal used in docs (emd.md)
  arma::vec t = arma::linspace<arma::vec>(0.0, 2 * arma::datum::pi, N);
  arma::vec signal =
      arma::sin((20 * t) % (1 + 0.2 * t)) +
      arma::square(t) +
      arma::sin(13 * t);

  arma::mat imfsEMD;
  arma::vec residueEMD;
  mlpack::EMD(signal, imfsEMD, residueEMD, 5, 10, 1e-3);

  arma::mat imfsEEMD;
  arma::vec residueEEMD;
  mlpack::EEMD(signal, imfsEEMD, residueEEMD, 1, 0.001, 5, 10, 1e-3);

  // Compare through their reconstructions
  arma::vec reconEMD = arma::sum(imfsEMD, 1) + residueEMD;
  arma::vec reconEEMD = arma::sum(imfsEEMD, 1) + residueEEMD;

  const double relReconDiff =
      arma::norm(reconEMD - reconEEMD, 2) / arma::norm(reconEMD, 2);
  UNSCOPED_INFO("Relative reconstruction difference"<< relReconDiff);
  REQUIRE(relReconDiff < 0.001);
}
