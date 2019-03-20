/**
 * @file mcc_impl.hpp
 * @author Gaurav Sharma
 *
 * Implementation of the class MCC.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_MCC_IMPL_HPP
#define MLPACK_CORE_CV_METRICS_MCC_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>

namespace mlpack {
namespace cv {

template<size_t PC /* PositiveClass */>
template<typename MLAlgorithm, typename DataType>
double MCC<PC>::Evaluate(MLAlgorithm& model,
                                   const DataType& data,
                                   const arma::Row<size_t>& labels)
{
  AssertSizes(data, labels, "MCC::Evaluate()");

  arma::Row<size_t> predictedLabels;
  model.Classify(data, predictedLabels);

  size_t tp = arma::sum((labels == PC) % (predictedLabels == PC));
  size_t tn = arma::sum((labels != PC) % (predictedLabels != PC));
  size_t fp = arma::sum((labels != PC) % (predictedLabels == PC));
  size_t fn = arma::sum((labels == PC) % (predictedLabels != PC));

  double mcc = (double) ((tp * tn) - (fp * fn)) /
               sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

  return mcc; 
}

} // namespace cv
} // namespace mlpack

#endif
