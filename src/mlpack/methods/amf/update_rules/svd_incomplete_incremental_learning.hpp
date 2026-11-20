/**
 * @file methods/amf/update_rules/svd_incomplete_incremental_learning.hpp
 * @author Sumedh Ghaisas
 *
 * SVD factorizer used in AMF (Alternating Matrix Factorization).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_SVD_INCOMPLETE_INCREMENTAL_LEARNING_HPP
#define MLPACK_METHODS_AMF_SVD_INCOMPLETE_INCREMENTAL_LEARNING_HPP

#include <mlpack/prereqs.hpp>
#include "incremental_iterators.hpp"

namespace mlpack {

/**
 * This class computes SVD using incomplete incremental batch learning, as
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
 * This class implements 'Algorithm 2' as given in the paper.  Incremental
 * learning modifies only some feature values in W and H after scanning part of
 * the input matrix (V).  This differs from batch learning, which considers
 * every element in V for each update of W and H.  The regularization technique
 * is also different: in incomplete incremental learning, regularization takes
 * into account the number of elements in a given column of V.
 *
 * @see SVDBatchLearning
 */
template<typename MatType>
class SVDIncompleteIncrementalLearning
{
 public:
  /**
   * Initialize the parameters of SVDIncompleteIncrementalLearning.
   *
   * @param u Step value used in batch learning.
   * @param kw Regularization constant for W matrix.
   * @param kh Regularization constant for H matrix.
   */
  SVDIncompleteIncrementalLearning(double u = 0.001,
                                   double kw = 0,
                                   double kh = 0)
          : u(u), kw(kw), kh(kh), currentUserIndex(0), currentItemIndex(0)
  {
    // Nothing to do.
  }

  /**
   * Initialize parameters before factorization.  This function must be called
   * before a new factorization.  This simply sets the column being considered
   * to 0, so the input matrix and rank are not used.
   *
   * @param * (dataset) Input matrix to be factorized.
   * @param * (rank) of factorization
   */
  void Initialize(const MatType& dataset, const size_t /* rank */)
  {
    InitializeVIter(dataset, vIter, currentUserIndex, currentItemIndex);
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
  template<typename WHMatType>
  inline void WUpdate(const MatType& V,
                      WHMatType& W,
                      const WHMatType& H)
  {
    WHMatType deltaW;
    deltaW.zeros(V.n_rows, W.n_cols);

    // Store an old copy of the position, because we will need to start from the
    // same place for the H update step when it is called.
    typename MatType::const_iterator vIterOld = vIter;
    const size_t oldUserIndex = currentUserIndex;
    const size_t oldItemIndex = currentItemIndex;

    // Iterate through all the rating by this user to update corresponding item
    // feature feature vector.
    size_t userIndex = currentUserIndex;
    while (currentUserIndex == userIndex)
    {
      const typename MatType::elem_type val = (*vIter);
      deltaW.row(currentItemIndex) +=
          (val - dot(W.row(currentItemIndex), H.col(currentUserIndex))) *
          H.col(currentUserIndex).t();

      // Add regularization.
      if (kw != 0)
        deltaW.row(currentItemIndex) -= kw * W.row(currentItemIndex);

      IncrementVIter(V, vIter, userIndex, currentItemIndex);
    }

    W += u * deltaW;

    // Restore position to old position for H update.
    vIter = vIterOld;
    currentUserIndex = oldUserIndex;
    currentItemIndex = oldItemIndex;
  }

  /**
   * The update rule for the encoding matrix H.  The function takes in all the
   * matrices and only changes the value of the H matrix.
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

    // Iterate through all the rating by this user to update corresponding item
    // feature feature vector.
    size_t userIndex = currentUserIndex;
    while (currentUserIndex == userIndex)
    {
      const typename MatType::elem_type val = (*vIter);
      deltaH += (val - dot(W.row(currentItemIndex), H.col(currentUserIndex))) *
          W.row(currentItemIndex).t();

      IncrementVIter(V, vIter, userIndex, currentItemIndex);
    }

    // Add regularization.
    if (kh != 0)
      deltaH -= kh * H.col(currentUserIndex);

    // Update H matrix and move on to the next user.
    H.col(currentUserIndex) += u * deltaH;
    currentUserIndex = userIndex;
  }

 private:
  //! Step size of batch learning.
  double u;
  //! Regularization parameter for W matrix.
  double kw;
  //! Regularization parameter for H matrix.
  double kh;

  //! Iterator pointing to the first nonzero element for the next user.
  typename MatType::const_iterator vIter;
  //! Current user under consideration.
  size_t currentUserIndex;
  //! First nonzero item for the given user under consideration.
  size_t currentItemIndex;
};

} // namespace mlpack

#endif
