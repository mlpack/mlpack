/**
 * @file svd_wrapper.hpp
 * @author Sumedh Ghaisas
 *
 * Wrapper class for SVD factorizers used for Collaborative Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SVDWRAPPER_HPP
#define MLPACK_METHODS_SVDWRAPPER_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace cf
{

/**
 * This class acts as a dummy class for passing as template parameter. Passing
 * this class as a template parameter to class SVDWrapper will force SVDWrapper
 * to use Armadillo's SVD implementation.
 */
class DummyClass {};

/**
 * This class acts as the wrapper for all SVD factorizers which are incompatible
 * with CF module. Normally SVD factrorizers implement Apply method which takes
 * matrix V and factorizes it into P, sigma and Q where V = P * sigma * trans(Q).
 * But CF module requires factrorization to be V = W * H. This class multiplies
 * P and sigma and takes the first 'r' eigenvectors out where 'r' is the rank
 * of factorization. Q matrix is transposed and trimmed to support the rank
 * of factorization. The Factroizer class should implement Apply which takes
 * matrices P, sigma, Q and V as their parameter respectively.
 */
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

//! add simple typedefs
typedef SVDWrapper<DummyClass> ArmaSVDFactorizer;

//! include the implementation
#include "svd_wrapper_impl.hpp"

} // namespace cf
} // namespace mlpack

#endif
