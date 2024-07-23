/**
 * @file methods/amf/update_rules/svd_complete_incremental_learning.hpp
 * @author Sumedh Ghaisas
 *
 * SVD factorizer used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_SVD_COMPLETE_INCREMENTAL_LEARNING_HPP
#define MLPACK_METHODS_AMF_SVD_COMPLETE_INCREMENTAL_LEARNING_HPP

#include <mlpack/prereqs.hpp>
#include "incremental_iterators.hpp"

namespace mlpack {

/**
 * This class computes SVD using complete incremental batch learning, as
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
 * This class implements 'Algorithm 3' given in the paper.  Complete incremental
 * learning is an extreme case of incremental learning, where feature vectors
 * are updated after looking at each single element in the input matrix (V).
 * This approach differs from incomplete incremental learning where feature
 * vectors are updated after seeing columns of elements in the input matrix.
 *
 * @see SVDIncompleteIncrementalLearning
 */
template <class MatType = arma::mat>
class SVDCompleteIncrementalLearning
{
 public:
  /**
   * Initialize the SVDCompleteIncrementalLearning class with the given
   * parameters.
   *
   * @param u Step value used in batch learning.
   * @param kw Regularization constant for W matrix.
   * @param kh Regularization constant for H matrix.
   */
  SVDCompleteIncrementalLearning(double u = 0.0001,
                                 double kw = 0,
                                 double kh = 0)
            : u(u), kw(kw), kh(kh), currentUserIndex(0), currentItemIndex(0)
  {
    // Nothing to do.
  }

  /**
   * Initialize parameters before factorization.  This function must be called
   * before a new factorization.  For this initialization, the input parameters
   * are unnecessary; we are only setting the current element index to 0.
   *
   * @param * (dataset) Input matrix to be factorized.
   * @param * (rank) Rank of factorization.
   */
  void Initialize(const MatType& dataset, const size_t /* rank */)
  {
    // Initialize the current score counters.
    InitializeVIter(dataset, vIter, currentUserIndex, currentItemIndex);
  }

  /**
   * The update rule for the basis matrix W.  The function takes in all the
   * matrices and only changes the value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  template<typename WHMatType>
  inline void WUpdate(const MatType& /* V */,
                      WHMatType& W,
                      const WHMatType& H)
  {
    WHMatType deltaW;
    deltaW.zeros(1, W.n_cols);

    const double val = (*vIter);

    // Update feature vector if current entry is non-zero and break the loop.
    deltaW += (val - dot(W.row(currentItemIndex),
        H.col(currentUserIndex))) * H.col(currentUserIndex).t();

    // Add regularization.
    if (kw != 0)
      deltaW -= kw * W.row(currentItemIndex);

    W.row(currentItemIndex) += u * deltaW;

    // We don't increment the iterator, as the H update needs to look at the
    // same element from V.
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
  template<typename WHMatType>
  inline void HUpdate(const MatType& V,
                      const WHMatType& W,
                      WHMatType& H)
  {
    WHMatType deltaH;
    deltaH.zeros(H.n_rows, 1);

    const double val = V(currentItemIndex, currentUserIndex);

    // Update H matrix based on the non-zero entry found in WUpdate function.
    deltaH += (val - dot(W.row(currentItemIndex),
        H.col(currentUserIndex))) * W.row(currentItemIndex).t();
    // Add regularization.
    if (kh != 0)
      deltaH -= kh * H.col(currentUserIndex);

    H.col(currentUserIndex) += u * deltaH;

    IncrementVIter(V, vIter, currentUserIndex, currentItemIndex);
  }

 private:
  //! Step count of batch learning.
  double u;
  //! Regularization parameter for matrix W.
  double kw;
  //! Regularization parameter for matrix H.
  double kh;

  //! Iterator pointing to the next nonzero element.
  typename MatType::const_iterator vIter;
  //! User of index of current entry.
  size_t currentUserIndex;
  //! Item index of current entry.
  size_t currentItemIndex;
};

} // namespace mlpack

#endif
