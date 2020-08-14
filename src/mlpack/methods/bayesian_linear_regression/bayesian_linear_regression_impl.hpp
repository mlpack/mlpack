/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_impl.hpp
 * @author Clement Mercier
 *
 * Implementation of templated BayesianLinearRegression functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP

#include "bayesian_linear_regression.hpp"

namespace mlpack {
namespace regression {

/**
 * Serialize the Bayesian linear regression model.
 */
template<typename Archive>
void BayesianLinearRegression::serialize(Archive& ar)
{
  ar & CEREAL_NVP(centerData);
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(scaleData);
  ar & CEREAL_NVP(nIterMax);
  ar & CEREAL_NVP(tol);
  ar & CEREAL_NVP(dataOffset);
  ar & CEREAL_NVP(dataScale);
  ar & CEREAL_NVP(responsesOffset);
  ar & CEREAL_NVP(alpha);
  ar & CEREAL_NVP(beta);
  ar & CEREAL_NVP(gamma);
  ar & CEREAL_NVP(omega);
  ar & CEREAL_NVP(matCovariance);
}

} // namespace regression
} // namespace mlpack

#endif
