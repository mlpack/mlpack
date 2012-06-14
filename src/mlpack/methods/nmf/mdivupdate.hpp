/**
 * @file mdivupdate.hpp
 * @author Mohan Rajendran
 *
 * Update rules for the Non-negative Matrix Factorization. This follows a method
 * described in the paper 'Algorithms for Non-negative Matrix Factorization' 
 * by D. D. Lee and H. S. Seung. This is a multiplicative rule that ensures
 * that the the 'divergence' 
 * \f$ \sum_i \sum_j (V_{ij} log\frac{V_{ij}}{(WH)_{ij}}-V_{ij}+(WH)_{ij}) \f$is
 * non-increasing between subsequent iterations. Both of the update rules
 * for W and H are defined in this file.
 *
 */

#ifndef __MLPACK_METHODS_NMF_MDIVUPDATE_HPP
#define __MLPACK_METHODS_NMF_MDIVUPDATE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nmf {

/**
 * The update rule for the basis matrix W. The formula used is 
 * \f[ 
 * W_{ia} \leftarrow W_{ia} \frac{\sum_{\mu} H_{a\mu} V_{i\mu}/(WH)_{i\mu}}
 * {\sum_{\nu} H_{a\nu}}
 * \f]
 */
class MultiplicativeDistanceW
{
 public:
  // Empty constructor required for the WUpdateRule template
  MultiplicativeDivergenceW() { }

  /**
   * The update function that actually updates the W matrix. The function takes
   * in all the salient matrices and only changes the value of the W matrix.
   *
   * @param V Input matrix to be factorized
   * @param W Basis matrix to be output
   * @param H Encoding matrix to output
   */

  inline static void Init(const arma::mat& V,
                     arma::mat& W, 
                     const arma::mat& H)
  {
    // Simple implementation. This can be left here.
    arma::mat t1;
    arma::rowvec t2;

    t1 = W*H;
    for(size_t i=0;i<W.n_rows;i++)
    {
      for(size_t j=0;j<W.n_cols;j++)
      {
        t2 = H.row(j)%V.row(i)/t1.row(i);
        W(i,j) = W(i,j)*sum(t2)/sum(H.row(i));
      }
    }

  }
}; // Class MultiplicativeDivergenceW

/**
 * The update rule for the encoding matrix H. The formula used is
 * \f[
 * H_{a\mu} \leftarrow H_{a\mu} \frac{\sum_{i} W_{ia} V_{i\mu}/(WH)_{i\mu}}
 * {\sum_{k} H_{ka}}
 * \f]
 */
class MultiplicativeDivergenceH
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
    arma::mat t1;
    arma::colvec t2;
    
    t1 = W*H;
    for(size_t i=0;i<H.n_rows;i++)
    {
      for(size_t j=0;j<H.n_cols;j++)
      {
        t2 = W.col(i)%V.col(j)/t1.col(j);
        H(i,j) = H(i,j)*sum(t2)/sum(H.col(i));
      }
    }

  }
}; // Class MultiplicativeDivergenceH

}; // namespace nmf
}; // namespace mlpack

#endif
