/**
 * @file mae_impl.hpp
 * @author Gaurav Sharma
 *
 * The implementation of the class MAE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_MAE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_MAE_IMPL_HPP

#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double MAE::Evaluate(MLAlgorithm& model,
                     const DataType& data,
                     const ResponsesType& responses)
{
  AssertColumnSizes(data, responses, "MAE::Evaluate()");

  ResponsesType predictedResponses;
  model.Predict(data, predictedResponses);
  double sum = arma::accu(arma::abs(responses - predictedResponses));

  return sum / responses.n_elem;
}

} // namespace cv
} // namespace mlpack

#endif
