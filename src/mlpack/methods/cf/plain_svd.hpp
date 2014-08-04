#ifndef __MLPACK_METHODS_PLAIN_SVD_HPP
#define __MLPACK_METHODS_PLAIN_SVD_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace svd
{

class PlainSVD
{
 public:
  PlainSVD() {};

  double Apply(const arma::mat& V,
               arma::mat& W,
               arma::mat& sigma,
               arma::mat& H) const;

  double Apply(const arma::mat& V,
               size_t r,
               arma::mat& W,
               arma::mat& H) const;
};

};
};

#endif
