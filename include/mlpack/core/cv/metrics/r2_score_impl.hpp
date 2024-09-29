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

template<bool AdjustedR2>
template<typename MLAlgorithm, typename DataType, typename ResponsesType>
double R2Score<AdjustedR2>::Evaluate(MLAlgorithm& model,
                                     const DataType& data,
                                     const ResponsesType& responses)
{
  util::CheckSameSizes(data, (size_t) responses.n_cols, "R2Score::Evaluate()",
      "responses");

  ResponsesType predictedResponses;
  // Taking Predicted Output from the model.
  model.Predict(data, predictedResponses);
  // Mean value of response.
  double meanResponses = arma::mean(responses);

  // Calculate the numerator i.e. residual sum of squares.
  double residualSumSquared = accu(arma::square(responses -
      predictedResponses));

  // Calculate the denominator i.e.total sum of squares.
  double totalSumSquared = accu(arma::square(responses - meanResponses));

  // Handling undefined R2 Score when both denominator and numerator is 0.0.
  if (residualSumSquared == 0.0)
    return totalSumSquared ? 1.0 : DBL_MIN;

  if (AdjustedR2)
  {
    // Returning adjusted R-squared.
    double rsq = 1 - (residualSumSquared / totalSumSquared);
    return (1 - ((1 - rsq) * ((data.n_cols - 1) /
        (data.n_cols - data.n_rows - 1))));
  }
  else
  {
    // Returning R-squared
    return 1 - residualSumSquared / totalSumSquared;
  }
}

} // namespace mlpack

#endif
