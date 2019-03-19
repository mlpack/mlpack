/**
 * @file r_squared_impl.hpp
 * @author Gaurav Sharma
 *
 * The implementation of the class RSquared.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_R_SQUARED_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_R_SQUARED_IMPL_HPP

#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double RSquared::Evaluate(MLAlgorithm& model,
                     const DataType& data,
                     const ResponsesType& responses)
{
  AssertColumnSizes(data, responses, "RSquared::Evaluate()");

  ResponsesType predictedResponses;
  model.Predict(data, predictedResponses);

  double mean_responses = arma::mean(responses);

  // calculate SSR
  double SSR = arma::accu(arma::square(predictedResponses - mean_responses));

  // calculate SST
  double SST = arma::accu(arma::square(responses - mean_responses));

  return SSR / SST;
}

} // namespace cv
} // namespace mlpack

#endif
