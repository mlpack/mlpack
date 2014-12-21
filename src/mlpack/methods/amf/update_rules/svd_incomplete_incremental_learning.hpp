#ifndef SVD_INCREMENTAL_LEARNING_HPP_INCLUDED
#define SVD_INCREMENTAL_LEARNING_HPP_INCLUDED

namespace mlpack
{
namespace amf
{
class SVDIncompleteIncrementalLearning
{
 public:
  SVDIncompleteIncrementalLearning(double u = 0.001,
                                   double kw = 0,
                                   double kh = 0)
          : u(u), kw(kw), kh(kh)
    {}

  template<typename MatType>
  void Initialize(const MatType& dataset, const size_t rank)
  {
    (void)rank;
  
    n = dataset.n_rows;
    m = dataset.n_cols;

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
 *
 * This file is part of MLPACK 1.0.9.
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
  template<typename MatType>
  inline void WUpdate(const MatType& V,
                      arma::mat& W,
                      const arma::mat& H)
  {
    arma::mat deltaW(n, W.n_cols);
    deltaW.zeros();
    for(size_t i = 0;i < n;i++)
    {
      double val;
      if((val = V(i, currentUserIndex)) != 0)
        deltaW.row(i) += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                         arma::trans(H.col(currentUserIndex));
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

    for(size_t i = 0;i < n;i++)
    {
      double val;
      if((val = V(i, currentUserIndex)) != 0)
        deltaH += (val - arma::dot(W.row(i), H.col(currentUserIndex))) *
                                                    arma::trans(W.row(i));
    }
    if(kh != 0) deltaH -= kh * H.col(currentUserIndex);

    H.col(currentUserIndex++) += u * deltaH;
    currentUserIndex = currentUserIndex % m;
  }

 private:
  double u;
  double kw;
  double kh;

  size_t n;
  size_t m;

  size_t currentUserIndex;
};

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

