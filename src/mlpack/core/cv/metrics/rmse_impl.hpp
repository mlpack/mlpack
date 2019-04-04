/**
 * @file rmse_impl.hpp
 * @author Gaurav Sharma
 *
 * The implementation of the class RMSE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_RMSE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_RMSE_IMPL_HPP

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double RMSE::Evaluate(MLAlgorithm& model,
                     const DataType& data,
                     const ResponsesType& responses)
{
  return sqrt(MSE::Evaluate(model, data, responses, "ERROR::Evaluate_RMSE()"));
}

} // namespace cv
} // namespace mlpack

#endif
