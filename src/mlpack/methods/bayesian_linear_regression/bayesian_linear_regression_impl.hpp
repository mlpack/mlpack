/**
 * @file bayesian_linear_regression_impl.hpp
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
void BayesianLinearRegression::serialize(Archive& ar,
                                         const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(centerData);
  ar & BOOST_SERIALIZATION_NVP(scaleData);
  ar & BOOST_SERIALIZATION_NVP(nIterMax);
  ar & BOOST_SERIALIZATION_NVP(tol);
  ar & BOOST_SERIALIZATION_NVP(dataOffset);
  ar & BOOST_SERIALIZATION_NVP(dataScale);
  ar & BOOST_SERIALIZATION_NVP(responsesOffset);
  ar & BOOST_SERIALIZATION_NVP(alpha);
  ar & BOOST_SERIALIZATION_NVP(beta);
  ar & BOOST_SERIALIZATION_NVP(gamma);
  ar & BOOST_SERIALIZATION_NVP(omega);
  ar & BOOST_SERIALIZATION_NVP(matCovariance);
}

} // namespace regression
} // namespace mlpack

#endif
