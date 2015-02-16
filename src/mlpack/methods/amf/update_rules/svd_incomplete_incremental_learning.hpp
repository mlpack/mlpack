/**
 * @file svd_incomplete_incremental_learning.hpp
 * @author Sumedh Ghaisas
 *
 * SVD factorizer used in AMF (Alternating Matrix Factorization).
 */
#ifndef SVD_INCREMENTAL_LEARNING_HPP_INCLUDED
#define SVD_INCREMENTAL_LEARNING_HPP_INCLUDED

namespace mlpack
{
namespace amf
{

/**
 * This class computes SVD by method incomplete incremental batch learning. 
 * This procedure is described in the paper 'A Guide to singular Value Decomposition' 
 * by Chih-Chao Ma. Class implements 'Algorithm 2' given in the paper. 
 * Incremental learning modifies only some feature values in W and H after 
 * scanning part of the training data. Incomplete incremental learning approach 
 * is different from batch learning as each object feature vector Mj has an 
 * additional regularization coefficient, which is equal to the number of 
 * existing scores for object j. Therefore, an object with more scores has a 
 * larger regularization coefficient in this incremental learning approach.
 *
 * @see SVDBatchLearning
 */
class SVDIncompleteIncrementalLearning
{
 public:
  /**
   * Empty constructor
   *
   * @param u step value used in batch learning
   * @param kw regularization constant for W matrix
   * @param kh regularization constant for H matrix
   */
  SVDIncompleteIncrementalLearning(double u = 0.001,
                                   double kw = 0,
                                   double kh = 0)
          : u(u), kw(kw), kh(kh)
  {}

  /**
   * Initialize parameters before factorization.
   * This function must be called before a new factorization.
   *
   * @param dataset Input matrix to be factorized.
   * @param rank rank of factorization
   */
  template<typename MatType>
  void Initialize(const MatType& dataset, const size_t rank)
  {
    (void)rank;
  
    n = dataset.n_rows;
    m = dataset.n_cols;

    // set the current user to 0
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
    arma::mat deltaW(n, W.n_cols);
    deltaW.zeros();
    
    // iterate through all the rating by this user to update corresponding
    // item feature feature vector
    for(size_t i = 0;i < n;i++)
    {
      double val;
      // update only if the rating is non-zero
      if((val = V(i, currentUserIndex)) != 0)
        deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                         arma::trans(H.col(currentUserIndex));
      // add regularization
      if(kw != 0) deltaW.row(i) -= kw * W.row(i);
    }

    W += u*deltaW;
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
    arma::mat deltaH(H.n_rows, 1);
    deltaH.zeros();
    
    // iterate through all the rating by this user to update corresponding
    // item feature feature vector
    for(size_t i = 0;i < n;i++)
    {
      double val;
      // update only if the rating is non-zero
      if((val = V(i, currentUserIndex)) != 0)
        deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                                    arma::trans(W.row(i));
    }
    // add regularization
    if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

    // update H matrix and move on to the next user
    H.col(currentUserIndex++) += u * deltaH;
    currentUserIndex = currentUserIndex % m;
  }

 private:
  //! step count of btach learning
  double u;
  //! regularization parameter for W matrix
  double kw;
  //! regularization parameter for H matrix
  double kh;

  //! number of items
  size_t n;
  //! number of users
  size_t m;

  //! current user under consideration 
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
  arma::mat deltaW(n, W.n_cols);
  deltaW.zeros();
  for(arma::sp_mat::const_iterator it = V.begin_col(currentUserIndex);
                                      it != V.end_col(currentUserIndex);it++)
  {
    double val = *it;
    size_t i = it.row();
    deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                         arma::trans(H.col(currentUserIndex));
    if(kw != 0) deltaW.row(i) -= kw * W.row(i);
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
    if((val = V(i, currentUserIndex)) != 0)
      deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                                    arma::trans(W.row(i));
  }
  if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

  H.col(currentUserIndex++) += u * deltaH;
  currentUserIndex = currentUserIndex % m;
}

}; // namepsace amf
}; // namespace mlpack


#endif // SVD_INCREMENTAL_LEARNING_HPP_INCLUDED

