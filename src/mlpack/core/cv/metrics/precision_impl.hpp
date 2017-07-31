/**
 * @file precision_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the class Precision.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_PRECISION_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_PRECISION_IMPL_HPP

#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<>
template<typename MLAlgorithm, typename DataType>
double Precision<Binary>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Precision<Binary>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tp = arma::sum(labels % predictedLabels);
  size_t numberOfPositivePredictions = arma::sum(predictedLabels);

  return double(tp) / numberOfPositivePredictions;
}

template<>
template<typename MLAlgorithm, typename DataType>
double Precision<Micro>::Evaluate(MLAlgorithm& model,
                                  const DataType& data,
                                  const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Precision<Micro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  size_t tpTotal = 0;
  for (size_t c = 0; c < numClasses; ++c)
    tpTotal += arma::sum((labels == c) % (predictedLabels == c));

  return double(tpTotal) / labels.n_elem;
}

template<>
template<typename MLAlgorithm, typename DataType>
double Precision<Macro>::Evaluate(MLAlgorithm& model,
                                  const DataType& data,
                                  const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Precision<Macro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  arma::vec precisions = arma::vec(numClasses);
  for (size_t c = 0; c < numClasses; ++c)
  {
    size_t tp = arma::sum((labels == c) % (predictedLabels == c));
    size_t numberOfPositivePredictions = arma::sum(predictedLabels == c);
    precisions(c) = double(tp) / numberOfPositivePredictions;
  }

  return arma::mean(precisions);
}

} // namespace cv
} // namespace mlpack

#endif
