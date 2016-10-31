/**
 * @file svd_incomplete_incremental_learning.hpp
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

namespace mlpack
{
namespace amf
{

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
          : u(u), kw(kw), kh(kh)
  {
    // Nothing to do.
  }

  /**
   * Initialize parameters before factorization.  This function must be called
   * before a new factorization.  This simply sets the column being considered
   * to 0, so the input matrix and rank are not used.
   *
   * @param dataset Input matrix to be factorized.
   * @param rank rank of factorization
   */
  template<typename MatType>
  void Initialize(const MatType& /* dataset */, const size_t /* rank */)
  {
    // Set the current user to 0.
    currentUserIndex = 0;
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
    arma::mat deltaW;
    deltaW.zeros(V.n_rows, W.n_cols);

    // Iterate through all the rating by this user to update corresponding item
    // feature feature vector.
    for (size_t i = 0; i < V.n_rows; ++i)
    {
      const double val = V(i, currentUserIndex);
      // Update only if the rating is non-zero.
      if (val != 0)
        deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
            H.col(currentUserIndex).t();
      // Add regularization.
      if (kw != 0)
        deltaW.row(i) -= kw * W.row(i);
    }

    W += u * deltaW;
  }

  /**
   * The update rule for the encoding matrix H.  The function takes in all the
   * matrices and only changes the value of the H matrix.
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
    arma::vec deltaH;
    deltaH.zeros(H.n_rows);

    // Iterate through all the rating by this user to update corresponding item
    // feature feature vector.
    for (size_t i = 0; i < V.n_rows; ++i)
    {
      const double val = V(i, currentUserIndex);
      // Update only if the rating is non-zero.
      if (val != 0)
        deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
            W.row(i).t();
    }
    // Add regularization.
    if (kh != 0)
      deltaH -= kh * H.col(currentUserIndex);

    // Update H matrix and move on to the next user.
    H.col(currentUserIndex++) += u * deltaH;
    currentUserIndex = currentUserIndex % V.n_cols;
  }

 private:
  //! Step size of batch learning.
  double u;
  //! Regularization parameter for W matrix.
  double kw;
  //! Regularization parameter for H matrix.
  double kh;

  //! Current user under consideration.
  size_t currentUserIndex;
};

//! TODO : Merge this template specialized function for sparse matrix using
//!        common row_col_iterator

//! template specialiazed functions for sparse matrices
template<>
inline void SVDIncompleteIncrementalLearning::
                                    WUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                          arma::mat& W,
                                                          const arma::mat& H)
{
  arma::mat deltaW(V.n_rows, W.n_cols);
  deltaW.zeros();
  for(arma::sp_mat::const_iterator it = V.begin_col(currentUserIndex);
                                      it != V.end_col(currentUserIndex);it++)
  {
    double val = *it;
    size_t i = it.row();
    deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                         arma::trans(H.col(currentUserIndex));
    if (kw != 0) deltaW.row(i) -= kw * W.row(i);
  }

  W += u*deltaW;
}

template<>
inline void SVDIncompleteIncrementalLearning::
                              HUpdate<arma::sp_mat>(const arma::sp_mat& V,
                                                    const arma::mat& W,
                                                    arma::mat& H)
{
  arma::mat deltaH(H.n_rows, 1);
  deltaH.zeros();

  for(arma::sp_mat::const_iterator it = V.begin_col(currentUserIndex);
                                        it != V.end_col(currentUserIndex);it++)
  {
    double val = *it;
    size_t i = it.row();
    if ((val = V(i, currentUserIndex)) != 0)
      deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                                    arma::trans(W.row(i));
  }
  if (kh != 0) deltaH -= kh * H.col(currentUserIndex);

  H.col(currentUserIndex++) += u * deltaH;
  currentUserIndex = currentUserIndex % V.n_cols;
}

} // namepsace amf
} // namespace mlpack

#endif
