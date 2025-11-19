#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include "catch.hpp"

#include <mlpack/methods/emd/emd_first_imf.hpp>
#include <mlpack/methods/emd/emd.hpp>
using namespace mlpack;
using namespace arma;

TEST_CASE("EMDToy", "[EMD]")
{
  arma::vec t = arma::linspace<arma::vec>(0.0, 1.0, 1000);
  arma::vec x = arma::sin(2 * arma::datum::pi * 5.0 * t)
              + 0.5 * arma::sin(2 * arma::datum::pi * 20.0 * t);

  arma::mat imfs;
  arma::vec r;
  emd::Emd(x, imfs, r);

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
  emd::Emd(x, imfs, r);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  for (uword k = 0; k < imfs.n_cols; ++k)
    UNSCOPED_INFO("IMF[" << k << "] L2 = " << arma::norm(imfs.col(k)));

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
  const bool ok = data::Load("nonstationary_signal_toy.csv", x, false);
  REQUIRE(ok); 

  arma::mat imfs;
  arma::vec r;
  emd::Emd(x, imfs, r);

  UNSCOPED_INFO("IMF count: " << imfs.n_cols);
  for (uword k = 0; k < imfs.n_cols; ++k)
    UNSCOPED_INFO("IMF[" << k << "] L2 = " << arma::norm(imfs.col(k)));

  arma::vec recon = r;
  for (uword k = 0; k < imfs.n_cols; ++k)
    recon += imfs.col(k);

  const double relErr = arma::norm(recon - x, 2) / arma::norm(x, 2);
  UNSCOPED_INFO("Reconstruction error: " << relErr);

  REQUIRE(relErr < 1e-3);
}