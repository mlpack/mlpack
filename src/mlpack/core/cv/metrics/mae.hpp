/**
 * @file mae.hpp
 * @author Gaurav Sharma
 *
 * The mean absolute error (MAE).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_MAE_HPP
#define MLPACK_CORE_CV_METRICS_MAE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * The MeanAbsoluteError is a metric of performance for regression algorithms
 * that is equal to the mean absolute error between predicted values and ground
 * truth (correct) values for given test items.
 */
class MAE
{
 public:
  /**
   * Run prediction and calculate the mean absolute error.
   *
   * @param model A regression model.
   * @param data Column-major data containing test items.
   * @param responses Ground truth (correct) target values for the test items,
   *     should be either a row vector or a column-major matrix.
   */
  template<typename MLAlgorithm, typename DataType, typename ResponsesType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& responses);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to minimize the measurement.
   */
  static const bool NeedsMinimization = true;
};

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "mae_impl.hpp"

#endif
