/**
 * @file plain_svd.hpp
 * @author Sumedh Ghaisas
 *
 * Wrapper class for Armadillo's SVD.
 */
#ifndef __MLPACK_METHODS_PLAIN_SVD_HPP
#define __MLPACK_METHODS_PLAIN_SVD_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace svd
{

/**
 * This class acts as a wrapper class for Armadillo's SVD implementation to be 
 * used by Collaborative Filteraing module.
 *
 * @see CF
 */
class PlainSVD
{
 public:
  // empty constructor
  PlainSVD() {};

  /**
   * Factorizer function which takes SVD of the given matrix and returns the 
   * frobenius norm of error.
   *
   * @param V input matrix
   * @param W first unitary matrix
   * @param sigma eigenvalue matrix
   * @param H second unitary matrix
   *
   * @note V = W * sigma * arma::trans(H)
   */
  double Apply(const arma::mat& V,
               arma::mat& W,
               arma::mat& sigma,
               arma::mat& H) const;
  /**
   * Factorizer function which computes SVD and returns matrices as required by 
   * CF module.
   * 
   * @param V input matrix
   * @param W first unitary matrix
   * @param H second unitary matrix
   *
   * @note V = W * H
   */
  double Apply(const arma::mat& V,
               size_t r,
               arma::mat& W,
               arma::mat& H) const;
}; // class PlainSVD

}; // namespace svd
}; // namespace mlpack

#endif
