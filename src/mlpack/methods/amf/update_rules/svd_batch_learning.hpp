/**
 * @file svd_batch_learning.hpp
 * @author Sumedh Ghaisas
 *
 * SVD factorizer used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
#define MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements SVD batch learning with momentum. This procedure is
 * described in the following paper:
 *
 * @code
 * @techreport{ma2008guide,
 *   title={A Guide to Singular Value Decomposition for Collaborative
 *       Filtering},
 *   author={Ma, Chih-Chao},
 *   year={2008},
 *   institution={Department of Computer Science, National Taiwan University}
 * }
 * @endcode
 *
 * This class implements 'Algorithm 4' as given in the paper.
 *
 * The factorizer decomposes the matrix V into two matrices W and H such that
 * sum of sum of squared error between V and W * H is minimum. This optimization
 * is performed with gradient descent. To make gradient descent faster, momentum
 * is added.
 */
class SVDBatchLearning
{
 public:
  /**
   * SVD Batch learning constructor.
   *
   * @param u step value used in batch learning
   * @param kw regularization constant for W matrix
   * @param kh regularization constant for H matrix
   * @param momentum momentum applied to batch learning process
   */
  SVDBatchLearning(double u = 0.0002,
                   double kw = 0,
                   double kh = 0,
                   double momentum = 0.9)
        : u(u), kw(kw), kh(kh), momentum(momentum)
  {
    // empty constructor
  }

  /**
   * Initialize parameters before factorization.  This function must be called
   * before a new factorization.  This resets the internally-held momentum.
   *
   * @param dataset Input matrix to be factorized.
   * @param rank rank of factorization
   */
  template<typename MatType>
  void Initialize(const MatType& dataset, const size_t rank)
  {
    const size_t n = dataset.n_rows;
    const size_t m = dataset.n_cols;

    mW.zeros(n, rank);
    mH.zeros(rank, m);
  }

  /**
   * The update rule for the basis matrix W.
   * The function takes in all the matrices and only changes the
   * value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename MatType>
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    size_t r = W.n_cols;

    // initialize the momentum of this iteration.
    mW = momentum * mW;

    // Compute the step.
    arma::mat deltaW;
    deltaW.zeros(n, r);
    for (size_t i = 0; i < n; i++)
    {
      for (size_t j = 0; j < m; j++)
      {
        const double val = V(i, j);
        if (val != 0)
          deltaW.row(i) += (val - arma::dot(W.row(i), H.col(j))) *
                                            arma::trans(H.col(j));
      }
      // Add regularization.
      if (kw != 0)
        deltaW.row(i) -= kw * W.row(i);
    }

    // Add the step to the momentum.
    mW += u * deltaW;
    // Add the momentum to the W matrix.
    W += mW;
  }

  /**
   * The update rule for the encoding matrix H.
   * The function takes in all the matrices and only changes the
   * value of the H matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix.
   * @param H Encoding matrix to be updated.
   */
  template<typename MatType>
  inline void HUpdate(const MatType& V,
                      const arma::mat& W,
                      arma::mat& H)
  {
    size_t n = V.n_rows;
    size_t m = V.n_cols;

    size_t r = W.n_cols;

    // Initialize the momentum of this iteration.
    mH = momentum * mH;

    // Compute the step.
    arma::mat deltaH;
    deltaH.zeros(r, m);
    for (size_t j = 0; j < m; j++)
    {
      for (size_t i = 0; i < n; i++)
      {
        const double val = V(i, j);
        if (val != 0)
          deltaH.col(j) += (val - arma::dot(W.row(i), H.col(j))) * W.row(i).t();
      }
      // Add regularization.
      if (kh != 0)
        deltaH.col(j) -= kh * H.col(j);
    }

    // Add this step to the momentum.
    mH += u * deltaH;
    // Add the momentum to H.
    H += mH;
  }

  //! Serialize the SVDBatch object.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using data::CreateNVP;
    ar & CreateNVP(u, "u");
    ar & CreateNVP(kw, "kw");
    ar & CreateNVP(kh, "kh");
    ar & CreateNVP(momentum, "momentum");
    ar & CreateNVP(mW, "mW");
    ar & CreateNVP(mH, "mH");
  }

 private:
  //! Step size of the algorithm.
  double u;
  //! Regularization parameter for matrix W.
  double kw;
  //! Regularization parameter for matrix H.
  double kh;
  //! Momentum value (between 0 and 1).
  double momentum;

  //! Momentum matrix for matrix W
  arma::mat mW;
  //! Momentum matrix for matrix H
  arma::mat mH;
}; // class SVDBatchLearning

//! TODO : Merge this template specialized function for sparse matrix using
//!        common row_col_iterator

/**
 * WUpdate function specialization for sparse matrix
 */
template<>
inline void SVDBatchLearning::WUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                    arma::mat& W,
                                                    const arma::mat& H)
{
  const size_t n = V.n_rows;
  const size_t r = W.n_cols;

  mW = momentum * mW;

  arma::mat deltaW;
  deltaW.zeros(n, r);

  for (arma::sp_mat::const_iterator it = V.begin(); it != V.end(); ++it)
  {
    const size_t row = it.row();
    const size_t col = it.col();
    deltaW.row(it.row()) += (*it - arma::dot(W.row(row), H.col(col))) *
                                             arma::trans(H.col(col));
  }

  if (kw != 0)
    deltaW -= kw * W;

  mW += u * deltaW;
  W += mW;
}

template<>
inline void SVDBatchLearning::HUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                    const arma::mat& W,
                                                    arma::mat& H)
{
  const size_t m = V.n_cols;
  const size_t r = W.n_cols;

  mH = momentum * mH;

  arma::mat deltaH;
  deltaH.zeros(r, m);

  for (arma::sp_mat::const_iterator it = V.begin(); it != V.end(); ++it)
  {
    const size_t row = it.row();
    const size_t col = it.col();
    deltaH.col(col) += (*it - arma::dot(W.row(row), H.col(col))) *
        W.row(row).t();
  }

  if (kh != 0)
    deltaH -= kh * H;

  mH += u * deltaH;
  H += mH;
}

} // namespace amf
} // namespace mlpack

#endif // MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
