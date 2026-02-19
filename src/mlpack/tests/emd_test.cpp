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
  const arma::uword N = 2000;
  arma::vec t = arma::linspace<arma::vec>(0.0, 1.0, N);
  arma::vec sin = arma::sin(2.0 * arma::datum::pi * 5.0 * t);
  arma::vec noise = 0.05 * arma::randn<arma::vec>(N);
  arma::vec x = sin + noise;
  arma::mat imfs;
  arma::vec r;

  // run emd
  // add timer to test
  const auto start = std::chrono::high_resolution_clock::now();
  mlpack::EMD(x, imfs, r, /*maxImfs=*/5, /*maxSiftIter=*/50, /*tol=*/1e-3);
  const auto stop = std::chrono::high_resolution_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        stop - start).count();
  UNSCOPED_INFO("EMD runtime for toy test (ms): " << ms);

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
  arma::vec x = arma::linspace<arma::vec>(0.0, 5.0, 500);

  arma::mat imfs;
  arma::vec r;
  mlpack::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols == 0);
  const double relResid = arma::norm(r - x, 2) / arma::norm(x, 2);
  REQUIRE(relResid < 1e-12);
}

//test templates to check for both float and double
TEMPLATE_TEST_CASE("EMDTemplateReconstruction", "[EMD]", float, double)
{
  using eT = TestType;

  arma::Col<eT> t = arma::linspace<arma::Col<eT>>(eT(0), eT(1), 800);
  arma::Col<eT> x = arma::sin(eT(2)*eT(arma::datum::pi)*eT(4)*t)
                  + eT(0.25) * arma::sin(eT(2)*eT(arma::datum::pi)*eT(14)*t);
  // x = sin(2*pi*4*t) + 0.25*sin(2*pi*14*t)

  arma::Mat<eT> imfs;
  arma::Col<eT> r;
  mlpack::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols >= 1);

  arma::Col<eT> recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Template reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-3);
}
