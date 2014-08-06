#include "plain_svd.hpp"

using namespace mlpack;
using namespace mlpack::svd;

double PlainSVD::Apply(const arma::mat& V,
                       arma::mat& W,
                       arma::mat& sigma,
                       arma::mat& H) const
{
  arma::vec E;
  arma::svd(W, E, H, V);

  sigma.zeros(V.n_rows, V.n_cols);

  for(size_t i = 0;i < sigma.n_rows && i < sigma.n_cols;i++)
    sigma(i, i) = E(i, 0);

  arma::mat V_rec = W * sigma * arma::trans(H);

  return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
}

double PlainSVD::Apply(const arma::mat& V,
                       size_t r,
                       arma::mat& W,
                       arma::mat& H) const
{
  if(r > V.n_rows || r > V.n_cols)
  {
    Log::Info << "Rank " << r << ", given for decomposition is invalid." << std::endl;
    r = (V.n_rows > V.n_cols) ? V.n_cols : V.n_rows;
    Log::Info << "Setting decomposition rank to " << r << std::endl;
  }

  arma::vec sigma;
  arma::svd(W, sigma, H, V);

  W = W.submat(0, 0, W.n_rows - 1, r - 1);
  H = H.submat(0, 0, H.n_cols - 1, r - 1);

  sigma = sigma.subvec(0, r - 1);

  W = W * arma::diagmat(sigma);
  
  H = arma::trans(H);

  arma::mat V_rec = W * H;

  return arma::norm(V - V_rec, "fro") / arma::norm(V, "fro");
}
