/**
 * @file linear_constraint_impl.hpp
 * @author Evan Patterson
 *
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_LINEAR_CONSTRAINT_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_LINEAR_CONSTRAINT_IMPL_HPP

#include "linear_constraint.hpp"

namespace mlpack {
namespace optimization {

// MatrixConstraint implementation

template <typename MatrixType>
arma::vec MatrixConstraint<MatrixType>::Evaluate(const arma::mat& x)
{
  const size_t m = _a.size();
  arma::vec vals = arma::vec(m);
  for (size_t i = 0; i < m; i++) {
    vals[i] = accu(_a[i] % x) - _b[i];
  }
  return vals;
}

template <typename MatrixType>
arma::vec MatrixConstraint<MatrixType>::EvaluateLR(const arma::mat& r)
{
  const size_t m = _a.size();
  arma::vec vals = arma::vec(m);
  for (size_t i = 0; i < m; i++) {
    vals[i] = accu((_a[i] * r) % r) - _b[i];
  }
  return vals;
}

template <typename MatrixType>
arma::mat MatrixConstraint<MatrixType>::GradientLR(const arma::mat& r,
                                                   const arma::vec& y)
{
  arma::mat s = arma::zeros<arma::mat>(r.n_rows, r.n_rows);
  for (size_t i = 0; _a.size(); i++) {
    s -= y[i] * _a[i];
  }
  return 2*s*r;
}

// LowRankConstraint implementation

template <typename MatrixType>
arma::vec LowRankConstraint<MatrixType>::Evaluate(const arma::mat& x)
{
  const size_t m = _a_vector.size();
  arma::vec vals = arma::vec(m);
  for (size_t i = 0; i < m; i++) {
    double val = 0;
    for (size_t j = 0; j < _a_vector[i].n_cols; j++) {
      arma::vec a = _a_vector[i].col(j);
      val += _a_weight[i][j] * dot(a, x*a);
    }
    vals[i] = val - _b[i];
  }
  return vals;
}

template <typename MatrixType>
arma::vec LowRankConstraint<MatrixType>::EvaluateLR(const arma::mat& r)
{
  const size_t m = _a_vector.size();
  arma::vec vals = arma::vec(m);
  for (size_t i = 0; i < m; i++) {
    double val = 0;
    for (size_t j = 0; j < _a_vector[i].n_cols; j++) {
      arma::vec a = _a_vector[i].col(j);
      val += _a_weight[i][j] * std::pow(arma::norm(trans(r)*a),2);
    }
    vals[i] = val - _b[i];
  }
  return vals;
}

template <typename MatrixType>
arma::mat LowRankConstraint<MatrixType>::GradientLR(const arma::mat& r,
                                                    const arma::vec& y)
{
  arma::mat grad = arma::zeros<arma::mat>(r.n_rows, r.n_cols);
  for (size_t i = 0; _a_vector.size(); i++) {
    for (size_t j = 0; j < _a_vector[i].n_cols; j++) {
      arma::vec a = _a_vector[i].col(j);
      grad -= ((y[i] * _a_weight[i][j]) * a) * (trans(a) * r);
    }
  }
  return 2*grad;
}

} // namespace optimization
} // namespace mlpack

#endif
