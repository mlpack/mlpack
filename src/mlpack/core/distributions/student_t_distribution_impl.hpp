/**
 * @file core/distributions/student_t_distribution_impl.hpp
 * @author Kiner Shah
 *
 * Implementation of the Student t-distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_STUDENT_T_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_STUDENT_T_DISTRIBUTION_IMPL_HPP

// In case it hasn't been included yet.
#include "student_t_distribution.hpp"

namespace mlpack {

/**
 * Return the log probability of the given observation under a univariate
 * Student t-distribution centered at the given location.
 */
template<typename MatType>
typename MatType::elem_type
StudentTDistribution<MatType>::LogProbability(
    const ElemType observation,
    const ElemType location) const
{
  // Log probability for univariate Student t-distribution:
  // log p(x | mu, df) = log(Gamma((df+1)/2)) - log(Gamma(df/2)) 
  //                     - 0.5 * log(df * pi) 
  //                     - ((df+1)/2) * log(1 + (x-mu)^2 / df)
  
  const ElemType diff = observation - location;
  const ElemType diffSquared = diff * diff;
  
  // Use lgamma for numerical stability
  const ElemType logNormalization = 
      std::lgamma((degreesOfFreedom + 1.0) / 2.0) -
      std::lgamma(degreesOfFreedom / 2.0) -
      0.5 * std::log(degreesOfFreedom * M_PI);
  
  const ElemType logKernel = 
      -((degreesOfFreedom + 1.0) / 2.0) * 
      std::log(1.0 + diffSquared / degreesOfFreedom);
  
  return logNormalization + logKernel;
}

/**
 * Compute the probability of pairwise distances under a multivariate
 * Student t-distribution (for t-SNE).
 */
template<typename MatType>
void StudentTDistribution<MatType>::PairwiseProbabilities(
    const MatType& squaredDistances,
    MatType& probabilities) const
{
  probabilities.set_size(squaredDistances.n_rows, squaredDistances.n_cols);
  
  // For t-SNE with 1 degree of freedom (Cauchy distribution):
  // q_ij = (1 + ||y_i - y_j||^2)^(-1)
  // More generally: (1 + ||y_i - y_j||^2 / df)^(-(df + 1)/2)
  
  const ElemType exponent = -(degreesOfFreedom + 1.0) / 2.0;
  
  for (size_t i = 0; i < squaredDistances.n_rows; ++i)
  {
    for (size_t j = 0; j < squaredDistances.n_cols; ++j)
    {
      probabilities(i, j) = std::pow(
          1.0 + squaredDistances(i, j) / degreesOfFreedom,
          exponent);
    }
  }
}

/**
 * Compute the log probability of pairwise distances under a multivariate
 * Student t-distribution.
 */
template<typename MatType>
void StudentTDistribution<MatType>::PairwiseLogProbabilities(
    const MatType& squaredDistances,
    MatType& logProbabilities) const
{
  logProbabilities.set_size(squaredDistances.n_rows, squaredDistances.n_cols);
  
  const ElemType logCoeff = -(degreesOfFreedom + 1.0) / 2.0;
  
  for (size_t i = 0; i < squaredDistances.n_rows; ++i)
  {
    for (size_t j = 0; j < squaredDistances.n_cols; ++j)
    {
      logProbabilities(i, j) = logCoeff * 
          std::log(1.0 + squaredDistances(i, j) / degreesOfFreedom);
    }
  }
}

} // namespace mlpack

#endif
