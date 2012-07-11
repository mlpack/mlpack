/**
 * @file alsupdate.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization. This follows a method
 * titled 'Alternating Least Squares' describes in the paper 'Positive Matrix
 * Factorization: A Non-negative Factor Model with Optimal Utilization of 
 * Error Estimates of Data Values' by P. Paatero and U. Tapper. It uses least 
 * squares projection formula to reduce the error value of 
 * \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ by alternately calculating W and H
 * respectively while holding the other matrix constant.
 *
 */

#ifndef __MLPACK_METHODS_NMF_ALSUPDATE_HPP
#define __MLPACK_METHODS_NMF_ALSUPDATE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

/**
 * The update rule for the basis matrix W. The formula used is 
 * \f[ 
 * W^T = \frac{HV^T}{HH^T}
 * \f]
 */
class AlternatingLeastSquareW
{
 public:
  // Empty constructor required for the WUpdateRule template
  AlternatingLeastSquareW() { }

  /**
   * The update function that actually updates the W matrix. The function takes
   * in all the salient matrices and only changes the value of the W matrix.
   *
   * @param V Input matrix to be factorized
   * @param W Basis matrix to be output
   * @param H Encoding matrix to output
   */

  inline static void Update(const arma::mat& V,
                     arma::mat& W, 
                     const arma::mat& H)
  {
    // Simple implementation. This can be left here.
    W = (inv(H*H.t())*H*H.t()).t();
    
    // Set all negative numbers to machine epsilon
    for(size_t i=0;i<W.n_rows*W.n_cols;i++)
    {
      if(W(i) < 0.0)
      {
        W(i) = eps(W);
      }
    }
  }
}; // Class AlternatingLeastSquareW

/**
 * The update rule for the encoding matrix H. The formula used is
 * \f[
 * H = \frac{W^TV}{W^TW}
 * \f]
 */
class AlternatingLeastSquareH
{
 public:
  // Empty constructor required for the HUpdateRule template
  AlternatingLeastSquareH() { }

  /**
   * The update function that actually updates the H matrix. The function takes
   * in all the salient matrices and only changes the value of the H matrix.
   *
   * @param V Input matrix to be factorized
   * @param W Basis matrix to be output
   * @param H Encoding matrix to output
   */

  inline static void Update(const arma::mat& V,
                     const arma::mat& W, 
                     arma::mat& H)
  {
    // Simple implementation. This can be left here.
    H = inv(W.t()*W)*W.t()*V;

    // Set all negative numbers to machine epsilon
    for(size_t i=0;i<H.n_rows*H.n_cols;i++)
    {
      if(H(i) < 0.0)
      {
        H(i) = eps(H);
      }
    }
  }
}; // Class AlternatingLeastSquareH

}; // namespace nmf
}; // namespace mlpack

#endif
