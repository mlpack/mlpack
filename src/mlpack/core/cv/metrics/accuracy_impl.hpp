/**
 * @file core/cv/metrics/accuracy_impl.hpp
 * @author Kirill Mishchenko
 *
 * The implementation of the class Accuracy.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ACCURACY_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_ACCURACY_IMPL_HPP

#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<typename MLAlgorithm, typename DataType>
double Accuracy::Evaluate(MLAlgorithm& model,
                          const DataType& data,
                          const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Accuracy::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);
  size_t amountOfCorrectPredictions = arma::sum(predictedLabels == labels);

  return (double) amountOfCorrectPredictions / labels.n_elem;
}

} // namespace cv
} // namespace mlpack

#endif
