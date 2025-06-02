/**
 * @file core/cv/metrics/facilities.hpp
 * @author Kirill Mishchenko
 * @author Khizir Siddiqui
 *
 * Functionality that is used more than in one metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_FACILITIES_HPP
#define MLPACK_CORE_CV_METRICS_FACILITIES_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/distances/lmetric.hpp>

namespace mlpack {

/**
  * Pairwise distance of the given data.
  *
  * @param data Column-major matrix.
  * @param metric Distance metric to be used.
  */
template<typename DataType, typename DistanceType>
DataType PairwiseDistances(const DataType& data,
                           const DistanceType& distance)
{
  DataType distances = DataType(data.n_cols, data.n_cols, arma::fill::none);
  for (size_t i = 0; i < data.n_cols; i++)
  {
    for (size_t j = 0; j < i; j++)
    {
      distances(i, j) = distance.Evaluate(data.col(i), data.col(j));
      distances(j, i) = distances(i, j);
    }
  }
  distances.diag().zeros();
  return distances;
}

} // namespace mlpack

#endif
