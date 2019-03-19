/**
 * @file adjusted_r_squared_impl.hpp
 * @author Gaurav Sharma
 *
 * The implementation of the class AdjustedRSquared.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ADJUSTED_R_SQUARED_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_ADJUSTED_R_SQUARED_IMPL_HPP

#include "r_squared.hpp"

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double AdjustedRSquared::Evaluate(MLAlgorithm& model,
                     const DataType& data,
                     const ResponsesType& responses)
{
  double r2 = RSquared::Evaluate(model, data, responses);

  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).
  const size_t data_points = data.n_cols;
  const size_t total_parameters = data.n_rows;

  double adj_r2 = 1 - ((1 - r2) * (data_points - 1) /
                  (data_points - total_parameters - 1));

  return adj_r2;
}

} // namespace cv
} // namespace mlpack

#endif
