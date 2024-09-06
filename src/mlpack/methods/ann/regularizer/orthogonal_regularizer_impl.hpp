/**
 * @file methods/ann/regularizer/orthogonal_regularizer_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of OrthogonalRegularizer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_IMPL_HPP
#define MLPACK_METHODS_ANN_ORTHOGONAL_REGULARIZER_IMPL_HPP

// In case it hasn't been included.
#include "orthogonal_regularizer.hpp"

namespace mlpack {

inline OrthogonalRegularizer::OrthogonalRegularizer(double factor) :
    factor(factor)
{
  // Nothing to do here.
}

template<typename MatType>
void OrthogonalRegularizer::Evaluate(const MatType& weight, MatType& gradient)
{
  arma::mat grad = zeros(arma::size(weight));

  for (size_t i = 0; i < weight.n_rows; ++i)
  {
    for (size_t j = 0; j < weight.n_rows; ++j)
    {
      if (i == j)
      {
        double s = arma::as_scalar(
            sign((weight.row(i) * weight.row(i).t()) - 1));
        grad.row(i) += 2 * s * weight.row(i);
      }
      else
      {
        double s = arma::as_scalar(sign(weight.row(i) * weight.row(j).t()));
        grad.row(i) += s * weight.row(j);
        grad.row(j) += s * weight.row(i);
      }
    }
  }

  gradient += vectorise(grad) * factor;
}

template<typename Archive>
void OrthogonalRegularizer::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(factor));
}

} // namespace mlpack

#endif
