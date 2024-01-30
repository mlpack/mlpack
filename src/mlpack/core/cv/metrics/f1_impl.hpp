/**
 * @file core/cv/metrics/f1_impl.hpp
 * @author Kirill Mishchenko
 *
 * Implementation of the class F1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_F1_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_F1_IMPL_HPP

#include <mlpack/core/cv/metrics/accuracy.hpp>

namespace mlpack {

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<typename MLAlgorithm, typename DataType>
double F1<AS, PC>::Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels)
{
  return Evaluate<AS>(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename>
double F1<AS, PC>::Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "F1<Binary>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tp = sum((labels == PC) % (predictedLabels == PC));
  size_t numberOfPositivePredictions = sum(predictedLabels == PC);
  size_t numberOfPositiveClassInstances = sum(labels == PC);

  double precision = double(tp) / numberOfPositivePredictions;
  double recall = double(tp) / numberOfPositiveClassInstances;

  return (precision + recall == 0.0) ? 0.0 :
      2.0 * precision * recall / (precision + recall);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename>
double F1<AS, PC>::Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "F1<Micro>::Evaluate()");

  // Microaveraged F1 is really the same as microaveraged precision and
  // microaveraged recall, which are in turn the same as accuracy.
  return Accuracy::Evaluate(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename, typename>
double F1<AS, PC>::Evaluate(MLAlgorithm& model,
                            const DataType& data,
                            const arma::Row<size_t>& labels)
{
  util::CheckSameSizes(data, labels, "F1<Macro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  arma::vec f1s = arma::vec(numClasses);
  for (size_t c = 0; c < numClasses; ++c)
  {
    size_t tp = sum((labels == c) % (predictedLabels == c));
    size_t positivePredictions = sum(predictedLabels == c);
    size_t positiveLabels = sum(labels == c);

    double precision = double(tp) / positivePredictions;
    double recall = double(tp) / positiveLabels;
    f1s(c) = (precision + recall == 0.0) ? 0.0 :
        2.0 * precision * recall / (precision + recall);
  }

  return arma::mean(f1s);
}

} // namespace mlpack

#endif
