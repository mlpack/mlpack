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

#include <mlpack/core/transforms/emd.hpp>
using namespace mlpack;
using namespace arma;


TEST_CASE("EMDSingleTone", "[EMD]")
{
  const arma::uword N = 2000;
  arma::vec t = arma::linspace<arma::vec>(0.0, 1.0, N);
  arma::vec x = arma::sin(2.0 * arma::datum::pi * 5.0 * t);

  arma::mat imfs;
  arma::vec r;

  // run emd
  const auto start = std::chrono::high_resolution_clock::now(); //add timer to test
  mlpack::EMD(x, imfs, r, /*maxImfs=*/5, /*maxSiftIter=*/25, /*tol=*/1e-3);
  const auto stop = std::chrono::high_resolution_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
  UNSCOPED_INFO("EMD runtime for toy test (ms): " << ms);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  REQUIRE(imfs.n_cols >= 1);

  const arma::vec imf0 = imfs.col(0);

  // IMF0 should match x , sin(5pi t).
  // check this with correlation constant
  const double c =
      std::abs(arma::as_scalar(arma::dot(imf0, x) /
              (arma::norm(imf0, 2) * arma::norm(x, 2))));
  UNSCOPED_INFO("abs(corr(IMF0,x))=" << c);
  REQUIRE(c > 0.99);

  // Residue should be small relative to the signal for toy
  const double relResid = arma::norm(r, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Residue relNorm=" << relResid);
  REQUIRE(relResid < 1e-2);

  // Reconstruction should be almost exact
  arma::vec recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

    // does the reconstruction match the original toy input
  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Reconstruction relErr=" << relErr);
  REQUIRE(relErr < 1e-6);
}

// TEST_CASE("EMDMonotoneNoImf", "[EMD]")
// {
//   // Monotone input should yield 0 IMFs and residue equals input. This guards
//   // the extrema endpoints logic, failure would mean there are
//   // IMFs when fewer than two extrema exist.
//   arma::vec x = arma::linspace<arma::vec>(0.0, 5.0, 500);

//   arma::mat imfs;
//   arma::vec r;
//   emd::EMD(x, imfs, r);

//   REQUIRE(imfs.n_cols == 0);
//   REQUIRE(arma::norm(r - x, 2) / arma::norm(x, 2) < 1e-12);
// }

// TEMPLATE_TEST_CASE("EMDTemplateReconstruction", "[EMD]", float, double)
// {
//   // Validate reconstruction accuracy for both float and double paths
//   // with the templated spline envelope wrapper.
//   using eT = TestType;
//   arma::Col<eT> t = arma::linspace<arma::Col<eT>>(eT(0), eT(1), 800);
//   arma::Col<eT> x = arma::sin(eT(2) * eT(arma::datum::pi) * eT(4) * t)
//                   + eT(0.25) * arma::sin(eT(2) * eT(arma::datum::pi) * eT(14) * t);

//   arma::Mat<eT> imfs;
//   arma::Col<eT> r;
//   emd::EMD(x, imfs, r);

//   // Extract at least one IMF and achieve accurate reconstruction.
//   REQUIRE(imfs.n_cols >= 1);

//   arma::Col<eT> recon = r;
//   for (arma::uword k = 0; k < imfs.n_cols; ++k)
//     recon += imfs.col(k);

//   const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
//   UNSCOPED_INFO("Template reconstruction relErr=" << relErr);
//   REQUIRE(relErr < 1e-3);
// }
