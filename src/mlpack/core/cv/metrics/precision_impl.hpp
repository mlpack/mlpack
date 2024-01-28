/**
 * @file core/cv/metrics/precision_impl.hpp
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

#include <mlpack/core/cv/metrics/accuracy.hpp>

namespace mlpack {

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<typename MLAlgorithm, typename DataType>
double Precision<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  return Evaluate<AS>(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename>
double Precision<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Precision<Binary>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tp = sum((labels == PC) % (predictedLabels == PC));
  size_t numberOfPositivePredictions = sum(predictedLabels == PC);

  return double(tp) / numberOfPositivePredictions;
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename>
double Precision<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Precision<Micro>::Evaluate()");

  // Microaveraged precision turns out to be just accuracy.
  return Accuracy::Evaluate(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename, typename>
double Precision<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Precision<Macro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  arma::vec precisions = arma::vec(numClasses);
  for (size_t c = 0; c < numClasses; ++c)
  {
    size_t tp = sum((labels == c) % (predictedLabels == c));
    size_t numberOfPositivePredictions = sum(predictedLabels == c);
    precisions(c) = double(tp) / numberOfPositivePredictions;
  }

  return arma::mean(precisions);
}

} // namespace mlpack

#endif
