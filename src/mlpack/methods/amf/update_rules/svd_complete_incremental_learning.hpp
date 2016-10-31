/**
 * @file svd_complete_incremental_learning.hpp
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

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{

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
template <class MatType>
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
            : u(u), kw(kw), kh(kh)
  {
    // Nothing to do.
  }

  /**
   * Initialize parameters before factorization.  This function must be called
   * before a new factorization.  For this initialization, the input parameters
   * are unnecessary; we are only setting the current element index to 0.
   *
   * @param dataset Input matrix to be factorized.
   * @param rank rank of factorization
   */
  void Initialize(const MatType& /* dataset */, const size_t /* rank */)
  {
    // Initialize the current score counters.
    currentUserIndex = 0;
    currentItemIndex = 0;
  }

  /**
   * The update rule for the basis matrix W.  The function takes in all the
   * matrices and only changes the value of the W matrix.
   *
   * @param V Input matrix to be factorized.
   * @param W Basis matrix to be updated.
   * @param H Encoding matrix.
   */
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    arma::mat deltaW;
    deltaW.zeros(1, W.n_cols);

    // Loop until a non-zero entry is found.
    while(true)
    {
      const double val = V(currentItemIndex, currentUserIndex);
      // Update feature vector if current entry is non-zero and break the loop.
      if (val != 0)
      {
        deltaW += (val - arma::dot(W.row(currentItemIndex),
            H.col(currentUserIndex))) * H.col(currentUserIndex).t();

        // Add regularization.
        if (kw != 0)
          deltaW -= kw * W.row(currentItemIndex);
        break;
      }
    }

    W.row(currentItemIndex) += u * deltaW;
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
  inline void HUpdate(const MatType& V,
                      const arma::mat& W,
                      arma::mat& H)
  {
    arma::mat deltaH;
    deltaH.zeros(H.n_rows, 1);

    const double val = V(currentItemIndex, currentUserIndex);

    // Update H matrix based on the non-zero entry found in WUpdate function.
    deltaH += (val - arma::dot(W.row(currentItemIndex),
        H.col(currentUserIndex))) * W.row(currentItemIndex).t();
    // Add regularization.
    if (kh != 0)
      deltaH -= kh * H.col(currentUserIndex);

    // Move on to the next entry.
    currentUserIndex = currentUserIndex + 1;
    if (currentUserIndex == V.n_rows)
    {
      currentUserIndex = 0;
      currentItemIndex = (currentItemIndex + 1) % V.n_cols;
    }

    H.col(currentUserIndex++) += u * deltaH;
  }

 private:
  //! Step count of batch learning.
  double u;
  //! Regularization parameter for matrix W.
  double kw;
  //! Regularization parameter for matrix H.
  double kh;

  //! User of index of current entry.
  size_t currentUserIndex;
  //! Item index of current entry.
  size_t currentItemIndex;
};

//! TODO : Merge this template specialized function for sparse matrix using
//!        common row_col_iterator

//! template specialiazed functions for sparse matrices
template<>
class SVDCompleteIncrementalLearning<arma::sp_mat>
{
  public:
  SVDCompleteIncrementalLearning(double u = 0.01,
                                 double kw = 0,
                                 double kh = 0)
            : u(u), kw(kw), kh(kh), it(NULL)
    {}

  ~SVDCompleteIncrementalLearning()
  {
    delete it;
  }

  void Initialize(const arma::sp_mat& dataset, const size_t rank)
  {
    (void)rank;
    n = dataset.n_rows;
    m = dataset.n_cols;

    it = new arma::sp_mat::const_iterator(dataset.begin());
    isStart = true;
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
  inline void WUpdate(const arma::sp_mat& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    if (!isStart) (*it)++;
    else isStart = false;

    if (*it == V.end())
    {
        delete it;
        it = new arma::sp_mat::const_iterator(V.begin());
    }

    size_t currentUserIndex = it->col();
    size_t currentItemIndex = it->row();

    arma::mat deltaW(1, W.n_cols);
    deltaW.zeros();

    deltaW += (**it - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex)))
                                      * arma::trans(H.col(currentUserIndex));
    if (kw != 0) deltaW -= kw * W.row(currentItemIndex);

    W.row(currentItemIndex) += u*deltaW;
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
  inline void HUpdate(const arma::sp_mat& V,
                      const arma::mat& W,
                      arma::mat& H)
  {
    (void)V;

    arma::mat deltaH(H.n_rows, 1);
    deltaH.zeros();

    size_t currentUserIndex = it->col();
    size_t currentItemIndex = it->row();

    deltaH += (**it - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex)))
                                        * arma::trans(W.row(currentItemIndex));
    if (kh != 0) deltaH -= kh * H.col(currentUserIndex);

    H.col(currentUserIndex) += u * deltaH;
  }

 private:
  double u;
  double kw;
  double kh;

  size_t n;
  size_t m;

  arma::sp_mat dummy;
  arma::sp_mat::const_iterator* it;

  bool isStart;
}; // class SVDCompleteIncrementalLearning

} // namespace amf
} // namespace mlpack

#endif

