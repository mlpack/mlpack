/**
 * @file specificity_impl.hpp
 * @author Gaurav Sharma
 *
 * Implementation of the class Specificity.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SPECIFICITY_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_SPECIFICITY_IMPL_HPP

#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<typename MLAlgorithm, typename DataType>
double Specificity<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  return Evaluate<AS>(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename>
double Specificity<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Specificity<Binary>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tn = arma::sum((labels != PC) % (predictedLabels != PC));
  size_t numberOfNegativeClassInstances = arma::sum(labels != PC);

  return double(tn) / numberOfNegativeClassInstances;
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename>
double Specificity<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Specificity<Micro>::Evaluate()");

  // Microaveraged specificty turns out to be just accuracy.
  return Accuracy::Evaluate(model, data, labels);
}

template<AverageStrategy AS, size_t PC /* PositiveClass */>
template<AverageStrategy _AS, typename MLAlgorithm, typename DataType, typename,
    typename, typename>
double Specificity<AS, PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "Specificity<Macro>::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t numClasses = arma::max(labels) + 1;

  arma::vec specificity = arma::vec(numClasses);
  for (size_t c = 0; c < numClasses; ++c)
  {
    size_t tn = arma::sum((labels != c) % (predictedLabels != c));
    size_t numberOfNegativeClassInstances = arma::sum(labels != c);
    specificity(c) = double(tn) / numberOfNegativeClassInstances;
  }

  return arma::mean(specificity);
}

} // namespace cv
} // namespace mlpack

#endif
