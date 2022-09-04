/**
 * @file core/cv/metrics/accuracy.hpp
 * @author Kirill Mishchenko
 *
 * The accuracy metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ACCURACY_HPP
#define MLPACK_CORE_CV_METRICS_ACCURACY_HPP

#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * The Accuracy is a metric of performance for classification algorithms that is
 * equal to a proportion of correctly labeled test items among all ones for
 * given test items.
 */
class Accuracy
{
 public:
  /**
   * Run classification and calculate accuracy.
   *
   * @param model A classification model.
   * @param data Column-major data containing test items.
   * @param labels Ground truth (correct) labels for the test items.
   */
  template<typename MLAlgorithm, typename DataType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const arma::Row<size_t>& labels);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to maximize the metric.
   */
  static const bool NeedsMinimization = false;
};

} // namespace mlpack

// Include implementation.
#include "accuracy_impl.hpp"

#endif
