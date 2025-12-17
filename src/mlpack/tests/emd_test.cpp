#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include "catch.hpp"


#include <mlpack/core/transforms/emd.hpp>
using namespace mlpack;
using namespace arma;

TEST_CASE("EMDToy", "[EMD]")
{
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

  REQUIRE(arma::norm(recon - x, 2) / arma::norm(x, 2) < 1e-6);
}

TEST_CASE("EMDDiagnosticToy", "[EMD]")
{
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

  // Reconstruction accuracy on mixed tones with slow trend.
  arma::vec recon = r;
  for (uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  double relErr = arma::norm(recon - x) / arma::norm(x);
  UNSCOPED_INFO("Reconstruction error: " << relErr);

  REQUIRE(relErr < 1e-3);
}

TEST_CASE("EMDDiagnostic", "[EMD]")
{
  vec x;
  REQUIRE(data::Load("nonstationary_signal_toy.csv", x, false));

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  for (uword k = 0; k < imfs.n_cols; ++k)
    UNSCOPED_INFO("IMF[" << k << "] L2 = " << arma::norm(imfs.col(k)));

  // Reconstruction accuracy on csv toy signal.
  arma::vec recon = r;
  for (uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Reconstruction error: " << relErr);

  REQUIRE(relErr < 1e-3);
}

TEST_CASE("EMDMonotoneNoImf", "[EMD]")
{
  // Monotone input should yield no IMFs and residue equals input.
  arma::vec x = arma::linspace<arma::vec>(0.0, 5.0, 500);

  arma::mat imfs;
  arma::vec r;
  emd::EMD(x, imfs, r);

  REQUIRE(imfs.n_cols == 0);
  REQUIRE(arma::norm(r - x, 2) / arma::norm(x, 2) < 1e-12);
}

TEMPLATE_TEST_CASE("EMDTemplateReconstruction", "[EMD]", float, double)
{
  // Validate reconstruction accuracy for both float and double paths.
  using eT = TestType;
  arma::Col<eT> t = arma::linspace<arma::Col<eT>>(eT(0), eT(1), 800);
  arma::Col<eT> x = arma::sin(eT(2) * eT(arma::datum::pi) * eT(4) * t)
                  + eT(0.25) * arma::sin(eT(2) * eT(arma::datum::pi) * eT(14) * t);

  arma::Mat<eT> imfs;
  arma::Col<eT> r;
  emd::EMD(x, imfs, r);

  // Check we extracted at least one IMF and can reconstruct.
  REQUIRE(imfs.n_cols >= 1);

  arma::Col<eT> recon = r;
  for (arma::uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  REQUIRE(relErr < 1e-3);
}
