/**
 * @file mcc.hpp
 * @author Gaurav Sharma
 *
 * The Matthews correlation coefficient metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_MCC_HPP
#define MLPACK_CORE_CV_METRICS_MCC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * Matthews correlation coefficient is a metric of performance for Binary
 * classification algorithms and is equal to
 * @f$ ((tp * tn) - (fp * fn)) /
 * sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) @f$,
 * where @f$ tp @f$, @f$ tn @f$, @f$ fp @f$ and @f$ fn @f$ are the
 * numbers of true positives, true negatives, false positives and
 * false negatives respectively.
 *
 * @tparam PositiveClass In the case of binary classification
 *         positives are assumed to have labels equal to this value.
 */
template<size_t PositiveClass = 1>
class MCC
{
 public:
  /**
   * Run classification and calculate mcc.
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

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "mcc_impl.hpp"

#endif
