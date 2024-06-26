/**
 * @file methods/amf/update_rules/svd_batch_learning.hpp
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

#include <mlpack/prereqs.hpp>

namespace mlpack {

// Forward declarations (implementations at end of file).
template<typename MatType, typename WHMatType>
void ComputeDeltaW(const MatType& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kw,
                   WHMatType& deltaW);

template<typename MatType, typename WHMatType>
void ComputeDeltaH(const MatType& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kh,
                   WHMatType& deltaH);

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
template<typename WHMatType = arma::mat>
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
                      WHMatType& W,
                      const WHMatType& H)
  {
    // initialize the momentum of this iteration.
    mW = momentum * mW;

    // Compute the step.
    WHMatType deltaW;
    ComputeDeltaW(V, W, H, kw, deltaW);

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
                      const WHMatType& W,
                      WHMatType& H)
  {
    // Initialize the momentum of this iteration.
    mH = momentum * mH;

    // Compute the step.
    WHMatType deltaH;
    ComputeDeltaH(V, W, H, kh, deltaH);

    // Add this step to the momentum.
    mH += u * deltaH;
    // Add the momentum to H.
    H += mH;
  }

  //! Serialize the SVDBatch object.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(u));
    ar(CEREAL_NVP(kw));
    ar(CEREAL_NVP(kh));
    ar(CEREAL_NVP(momentum));
    ar(CEREAL_NVP(mW));
    ar(CEREAL_NVP(mH));
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
  WHMatType mW;
  //! Momentum matrix for matrix H
  WHMatType mH;
}; // class SVDBatchLearning

template<typename MatType, typename WHMatType>
void ComputeDeltaW(const MatType& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kw,
                   WHMatType& deltaW)
{
  const size_t n = (size_t) V.n_rows;
  const size_t m = (size_t) V.n_cols;
  const size_t r = (size_t) W.n_cols;

  deltaW.zeros(n, r);
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < m; ++j)
    {
      const double val = V(i, j);
      if (val != 0)
        deltaW.row(i) += (val - dot(W.row(i), H.col(j))) * trans(H.col(j));
    }
    // Add regularization.
    if (kw != 0)
      deltaW.row(i) -= kw * W.row(i);
  }
}

// Specialization for sparse matrices: don't iterate over zero-valued elements.
template<typename eT, typename WHMatType>
void ComputeDeltaW(const arma::SpMat<eT>& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kw,
                   WHMatType& deltaW)
{
  const size_t n = (size_t) V.n_rows;
  const size_t m = (size_t) V.n_cols;
  const size_t r = (size_t) W.n_cols;

  deltaW.zeros(n, r);

  typename arma::SpMat<eT>::const_iterator it = V.begin();
  while (it != V.end())
  {
    const size_t row = it.row();
    const size_t col = it.col();
    deltaW.row(it.row()) += (*it - dot(W.row(row), H.col(col))) *
        trans(H.col(col));

    ++it;
  }

  if (kw != 0)
    deltaW -= kw * W;
}

template<typename MatType, typename WHMatType>
void ComputeDeltaH(const MatType& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kh,
                   WHMatType& deltaH)
{
  const size_t n = (size_t) V.n_rows;
  const size_t m = (size_t) V.n_cols;
  const size_t r = (size_t) W.n_cols;

  deltaH.zeros(r, m);
  for (size_t j = 0; j < m; ++j)
  {
    for (size_t i = 0; i < n; ++i)
    {
      const double val = V(i, j);
      if (val != 0)
        deltaH.col(j) += (val - dot(W.row(i), H.col(j))) * W.row(i).t();
    }
    // Add regularization.
    if (kh != 0)
      deltaH.col(j) -= kh * H.col(j);
  }
}

// Specialization for sparse matrices: don't iterate over zero-valued elements.
template<typename eT, typename WHMatType>
void ComputeDeltaH(const arma::SpMat<eT>& V,
                   const WHMatType& W,
                   const WHMatType& H,
                   const double kh,
                   WHMatType& deltaH)
{
  const size_t n = (size_t) V.n_rows;
  const size_t m = (size_t) V.n_cols;
  const size_t r = (size_t) W.n_cols;

  deltaH.zeros(r, m);

  typename arma::SpMat<eT>::const_iterator it = V.begin();
  while (it != V.end())
  {
    const size_t row = it.row();
    const size_t col = it.col();
    deltaH.col(col) += (*it - dot(W.row(row), H.col(col))) * W.row(row).t();

    ++it;
  }

  if (kh != 0)
    deltaH -= kh * H;
}

} // namespace mlpack

#endif // MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCH_LEARNING_HPP
