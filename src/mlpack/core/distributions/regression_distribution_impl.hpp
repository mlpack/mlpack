/**
 * @file core/distributions/regression_distribution_impl.hpp
 * @author Michael Fox
 *
 * Implementation of conditional Gaussian distribution for HMM regression
 * (HMMR).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_IMPL_HPP
#define MLPACK_CORE_DISTRIBUTIONS_REGRESSION_DISTRIBUTION_IMPL_HPP

#include "regression_distribution.hpp"

namespace mlpack {

/**
 * Estimate parameters using provided observation weights.
 *
 * @param observations List of observations.
 */
template<typename MatType>
inline void RegressionDistribution<MatType>::Train(const MatType& observations)
{
  LinearRegression<MatType> lr(observations.rows(1, observations.n_rows - 1),
      RowType(observations.row(0)), 0, true);
  rf = lr;
  RowType fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted);
}

template<typename MatType>
inline void RegressionDistribution<MatType>::Train(
    const MatType& observations,
    const RowType& weights)
{
  LinearRegression<MatType> lr(observations.rows(1, observations.n_rows - 1),
      RowType(observations.row(0)), weights, 0, true);
  rf = lr;
  RowType fitted;
  lr.Predict(observations.rows(1, observations.n_rows - 1), fitted);
  err.Train(observations.row(0) - fitted, weights.t());
}

/**
 * Evaluate probability density function of given observation.
 *
 * @param observation Point to evaluate probability at.
 */
template<typename MatType>
inline typename RegressionDistribution<MatType>::ElemType
RegressionDistribution<MatType>::Probability(const VecType& observation) const
{
  RowType fitted;
  rf.Predict(observation.rows(1, observation.n_rows - 1), fitted);
  return err.Probability(observation(0) - fitted.t());
}

/**
 * Evaluate probability density function on the given observations.
 *
 * @param observation Points to evaluate probability at.
 * @param probabilities Vector to store computed probabilities in.
 */
template<typename MatType>
inline void RegressionDistribution<MatType>::Probability(
    const MatType& observations,
    VecType& probabilities) const
{
  probabilities.set_size(observations.n_cols);
  for (size_t i = 0; i < observations.n_cols; ++i)
    probabilities[i] = Probability(observations.unsafe_col(i));
}

template<typename MatType>
inline void RegressionDistribution<MatType>::Predict(
    const MatType& points,
    RowType& predictions) const
{
  rf.Predict(points, predictions);
}

} // namespace mlpack

#endif
