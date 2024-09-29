/**
 * @file core/cv/metrics/recall_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the class Recall.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_RECALL_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_RECALL_IMPL_HPP

#include <mlpack/core/cv/metrics/accuracy.hpp>

namespace mlpack {

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<typename MLAlgorithm, typename DataType>
double Recall<AS, PC>::Evaluate(MLAlgorithm& model,
                                const DataType& data,
                                const arma::Row<size_t>& labels)
{
  return Evaluate<AS>(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename>
double Recall<AS, PC>::Evaluate(MLAlgorithm& model,
                                const DataType& data,
                                const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Recall<Binary>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tp = sum((labels == PC) % (predictedLabels == PC));
  size_t numberOfPositiveClassInstances = sum(labels == PC);

  return double(tp) / numberOfPositiveClassInstances;
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename>
double Recall<AS, PC>::Evaluate(MLAlgorithm& model,
                                const DataType& data,
                                const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Recall<Micro>::Evaluate()");

  // Microaveraged recall is really the same as accuracy.
  return Accuracy::Evaluate(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename, typename>
double Recall<AS, PC>::Evaluate(MLAlgorithm& model,
                                const DataType& data,
                                const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "Recall<Macro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  arma::vec recalls = arma::vec(numClasses);
  for (size_t c = 0; c < numClasses; ++c)
  {
    size_t tp = sum((labels == c) % (predictedLabels == c));
    size_t positiveLabels = sum(labels == c);
    recalls(c) = double(tp) / positiveLabels;
  }

  return arma::mean(recalls);
}

} // namespace mlpack

#endif
