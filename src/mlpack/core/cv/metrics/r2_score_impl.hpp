/**
 * @file core/cv/metrics/r2_score_impl.hpp
 * @author Bisakh Mondal
 *
 * The implementation of the class R2Score.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_R2SCORE_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_R2SCORE_IMPL_HPP

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double R2Score::Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& responses)
{
  if (data.n_cols != responses.n_cols)
  {
    std::ostringstream oss;
    oss << "R2Score::Evaluate(): number of points (" << data.n_cols << ") "
        << "does not match number of responses (" << responses.n_cols << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

  ResponsesType predictedResponses;
  // Taking Predicted Output from the model.
  model.Predict(data, predictedResponses);
  // Mean value of response.
  double meanResponses = arma::mean(responses);

  // Calculate the numerator i.e. residual sum of squares.
  double residualSumSquared = arma::accu(arma::square(responses -
      predictedResponses));

  // Calculate the denominator i.e.total sum of squares.
  double totalSumSquared = arma::accu(arma::square(responses - meanResponses));

  // Handling undefined R2 Score when both denominator and numerator is 0.0.
  if (residualSumSquared == 0.0)
    return totalSumSquared ? 1.0 : DBL_MIN;

  return 1 - residualSumSquared / totalSumSquared;
}

} // namespace cv
} // namespace mlpack

#endif
