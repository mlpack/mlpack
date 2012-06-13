/**
 * @file mdistupdate.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization. This follows a method
 * described in the paper 'Algorithms for Non-negative Matrix Factorization' 
 * by D. D. Lee and H. S. Seung. This is a multiplicative rule that ensures
 * that the Frobenius norm \f$ \sqrt{\sum_i \sum_j(V-WH)^2} \f$ is
 * non-increasing between subsequent iterations. Both of the update rules
 * for W and H are defined in this file.
 *
 */

#ifndef __MLPACK_METHODS_NMF_MDISTUPDATE_HPP
#define __MLPACK_METHODS_NMF_MDISTUPDATE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

/**
 * The update rule for the basis matrix W. The formula used is 
 * \f[ 
 * W_{ia} \leftarrow W_{ia} \frac{(VH^T)_{ia}}{(WHH^T)_{ia}}
 * \f]
 */
class MultiplicativeDistanceW
{
 public:
  // Empty constructor required for the WUpdateRule template
  MultiplicativeDistanceW() { }

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
    arma::mat t1,t2;
    
    t1 = V*H.t();
    t2 = W*H*H.t();
    
    W = (W%t1)/t2;
  }
}; // Class MultiplicativeDistanceW

/**
 * The update rule for the encoding matrix H. The formula used is
 * \f[
 * H_{a\mu} \leftarrow H_{a\mu} \frac{(W^T V)_{a\mu}}{(W^T WH)_{a\mu}}
 * \f]
 */
class MultiplicativeDistanceH
{
 public:
  // Empty constructor required for the HUpdateRule template
  MultiplicativeDistanceH() { }

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
    arma::mat t1,t2;
    
    t1 = W.t()*V;
    t2 = W.t()*W*H;

    H = (H%t1)/t2;
  }
}; // Class MultiplicativeDistanceH

}; // namespace nmf
}; // namespace mlpack

#endif
