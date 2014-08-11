/**
 * @file svd_wrapper.hpp
 * @author Sumedh Ghaisas
 *
 * Wrapper class for SVD factorizers used for Collaborative Filtering.
 */
#ifndef __MLPACK_METHODS_SVDWRAPPER_HPP
#define __MLPACK_METHODS_SVDWRAPPER_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace cf
{

/**
 *
 * @see CF
 */

class DummyClass {}; 
 
template<class Factorizer = DummyClass>
class SVDWrapper
{
 public:
  // empty constructor
  SVDWrapper(const Factorizer& factorizer = Factorizer()) 
    : factorizer(factorizer) {};

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
               
 private:
  //! svd factorizer
  Factorizer factorizer;
}; // class SVDWrapper

//! include the implementation
#include "svd_wrapper_impl.hpp"

}; // namespace cf
}; // namespace mlpack

#endif
