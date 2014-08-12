/**
 * @file svd_complete_incremental_learning.hpp
 * @author Sumedh Ghaisas
 *
 * SVD factorizer used in AMF (Alternating Matrix Factorization).
 */
#ifndef _MLPACK_METHODS_AMF_SVDCOMPLETEINCREMENTALLEARNING_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SVDCOMPLETEINCREMENTALLEARNING_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{

/**
 * This class computes SVD by method complete incremental batch learning. 
 * This procedure is described in the paper 'A Guide to singular Value Decomposition' 
 * by Chih-Chao Ma. Class implements 'Algorithm 3' given in the paper. Complete
 * incremental learning is an extreme extreme case of incremental learning where 
 * feature vectors are updated after looking at each single score. This approach
 * differs from incomplete incremental learning where feature vectors are updated 
 * after seeing scores of individual users.
 *
 * @see SVDIncompleteIncrementalLearning
 */
template <class MatType>
class SVDCompleteIncrementalLearning
{
 public:
  /**
   * Empty constructor
   *
   * @param u step value used in batch learning
   * @param kw regularization constant for W matrix
   * @param kh regularization constant for H matrix
   */
  SVDCompleteIncrementalLearning(double u = 0.0001,
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
  void Initialize(const MatType& dataset, const size_t rank)
  {
    (void)rank;
    n = dataset.n_rows;
    m = dataset.n_cols;

    // initialize the current score counters
    currentUserIndex = 0;
    currentItemIndex = 0;
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
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    arma::mat deltaW(1, W.n_cols);
    deltaW.zeros();
    
    // loop till a non-zero entry is found 
    while(true)
    {
      double val;
      // update feature vector if current entry is non-zero and break the loop
      if((val = V(currentItemIndex, currentUserIndex)) != 0)
      {
        deltaW += (val - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex))) 
                                        * arma::trans(H.col(currentUserIndex));
        // add regularization                               
        if(kw != 0) deltaW -= kw * W.row(currentItemIndex);
        break;
      }
    }

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
  inline void HUpdate(const MatType& V,
                      const arma::mat& W,
                      arma::mat& H)
  {
    arma::mat deltaH(H.n_rows, 1);
    deltaH.zeros();
    
    const double& val = V(currentItemIndex, currentUserIndex);
    
    // update H matrix based on the non-zero enrty found in WUpdate function
    deltaH += (val - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex))) 
                                      * arma::trans(W.row(currentItemIndex));
    // add regularization
    if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

    // move on to the next entry
    currentUserIndex = currentUserIndex + 1;
    if(currentUserIndex == n)
    {
      currentUserIndex = 0;
      currentItemIndex = (currentItemIndex + 1) % m;
    }

    H.col(currentUserIndex++) += u * deltaH;
  }

 private:
  //! step count of batch learning
  double u;
  //! regularization parameter for matrix w
  double kw;
  //! regualrization matrix for matrix H
  double kh;

  //! number of items
  size_t n;
  //! number of users
  size_t m;
  
  //! user of index of current entry
  size_t currentUserIndex;
  //! item index of current entry
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
    if(!isStart) (*it)++;
    else isStart = false;

    if(*it == V.end())
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
    if(kw != 0) deltaW -= kw * W.row(currentItemIndex);

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
    if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

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

}; // namespace amf
}; // namespace mlpack


#endif // _MLPACK_METHODS_AMF_SVDCOMPLETEINCREMENTALLEARNING_HPP_INCLUDED

