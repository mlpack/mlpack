/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 */
#ifndef __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP
#define __MLPACK_METHODS_AMF_UPDATE_RULES_SVD_BATCHLEARNING_HPP

#include <mlpack/core.hpp>

namespace mlpack
{
namespace amf
{
class SVDBatchLearning
{
 public:
  SVDBatchLearning(double u = 0.0002,
                   double kw = 0,
                   double kh = 0,
                   double momentum = 0.5,
                   double min = -DBL_MIN,
                   double max = DBL_MAX)
        : u(u), kw(kw), kh(kh), min(min), max(max), momentum(momentum)
    {}

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

    size_t r = W.n_cols;

    mW = momentum * mW;

    arma::mat deltaW(n, r);
    deltaW.zeros();

    for(typename MatType::const_iterator it = V.begin();it != V.end();it++)
    {
      size_t row = it.row();
      size_t col = it.col();
      deltaW.row(it.row()) += (*it - arma::dot(W.row(row), H.col(col))) * 
                                                    arma::trans(H.col(col));
    }

    if(kw != 0) for(size_t i = 0; i < n; i++)
    {
      deltaW.row(i) -= kw * W.row(i);
    }

    mW += u * deltaW;
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
    size_t m = V.n_cols;

    size_t r = W.n_cols;

    mH = momentum * mH;

    arma::mat deltaH(r, m);
    deltaH.zeros();

    for(typename MatType::const_iterator it = V.begin();it != V.end();it++)
    {
      size_t row = it.row();
      size_t col = it.col();
      deltaH.col(col) += (*it - arma::dot(W.row(row), H.col(col))) * 
                                                    arma::trans(W.row(row));
    }

    if(kh != 0) for(size_t j = 0; j < m; j++)
    {
      deltaH.col(j) -= kh * H.col(j);
    }

    mH += u*deltaH;
    H += mH;
  }
  
 private:
  double Predict(const arma::mat& wi, const arma::mat& hj) const
  {
    arma::mat temp = (wi * hj);
    double out = temp(0,0);
    return out;
  }

  double u;
  double kw;
  double kh;
  double min;
  double max;
  double momentum;

  arma::mat mW;
  arma::mat mH;
};
} // namespace amf
} // namespace mlpack


#endif


