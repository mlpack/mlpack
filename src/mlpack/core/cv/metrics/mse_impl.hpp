/**
 * @file core/cv/metrics/mse_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class MSE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_MSE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_MSE_IMPL_HPP

namespace mlpack {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double MSE::Evaluate(MLAlgorithm& model,
                     const DataType& data,
                     const ResponsesType& responses)
{
  util::CheckSameSizes(data, (size_t) responses.n_cols, "MSE::Evaluate()",
      "responses");

  ResponsesType predictedResponses;
  model.Predict(data, predictedResponses);
  double sum = accu(arma::square(responses - predictedResponses));

  return sum / responses.n_elem;
}

} // namespace mlpack

#endif
