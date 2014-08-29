#ifndef SVD_COMPLETE_INCREMENTAL_LEARNING_HPP_INCLUDED
#define SVD_COMPLETE_INCREMENTAL_LEARNING_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{

template <class MatType>
class SVDCompleteIncrementalLearning
{
 public:
  SVDCompleteIncrementalLearning(double u = 0.0001,
                                 double kw = 0,
                                 double kh = 0)
            : u(u), kw(kw), kh(kh)
    {}

  void Initialize(const MatType& dataset, const size_t rank)
  {
    (void)rank;
    n = dataset.n_rows;
    m = dataset.n_cols;

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
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
   */
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    arma::mat deltaW(1, W.n_cols);
    deltaW.zeros();
    while(true)
    {
      double val;
      if((val = V(currentItemIndex, currentUserIndex)) != 0)
      {
        deltaW += (val - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex))) 
                                        * arma::trans(H.col(currentUserIndex));
        if(kw != 0) deltaW -= kw * W.row(currentItemIndex);
        break;
      }
      currentUserIndex = currentUserIndex + 1;
      if(currentUserIndex == n)
      {
        currentUserIndex = 0;
        currentItemIndex = (currentItemIndex + 1) % m;
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

    while(true)
    {
      double val;
      if((val = V(currentItemIndex, currentUserIndex)) != 0)
      deltaH += (val - arma::dot(W.row(currentItemIndex), H.col(currentUserIndex))) 
                                      * arma::trans(W.row(currentItemIndex));
      if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

      currentUserIndex = currentUserIndex + 1;
      if(currentUserIndex == n)
      {
        currentUserIndex = 0;
        currentItemIndex = (currentItemIndex + 1) % m;
      }
    }

    H.col(currentUserIndex++) += u * deltaH;
  }

 private:
  double u;
  double kw;
  double kh;

  size_t n;
  size_t m;

  size_t currentUserIndex;
  size_t currentItemIndex;
};

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

    H.col(currentUserIndex++) += u * deltaH;
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
};

}
}


#endif // SVD_COMPLETE_INCREMENTAL_LEARNING_HPP_INCLUDED

