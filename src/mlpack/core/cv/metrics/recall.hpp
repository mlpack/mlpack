/**
 * @file recall.hpp
 * @author Kirill Mishchenko
 *
 * The recall metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_RECALL_HPP
#define MLPACK_CORE_CV_METRICS_RECALL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/average_strategy.hpp>

namespace mlpack {
namespace cv {

/**
 * Recall is a metric of performance for classification algorithms that for
 * binary classification is equal to tp / (tp + fn), where tp and fn are the
 * numbers of true positives and false negatives respectively. For multiclass
 * classification the recall metric can be used with the following strategies
 * for averaging.
 * 1. Micro. If there are N + 1 classes in total, the result is equal to
 * (tp0 + tp1 + ... + tpN) / (tp0 + tp1 + ... + tpN + fn0 + fn1 + ... fnN),
 * where tpI and fnI are the numbers of true positives and false negatives
 * respectively for the class (label) I.
 * 2. Macro. If there are N + 1 classes in total, the result is equal to the
 * mean of values tp0 / (tp0 + fn0), tp1 / (tp1 + fn1), ..., tpN / (tpN + fnN),
 * where tpI and fnI are the numbers of true positives and false negatives
 * respectively for the class (label) I.
 *
 * In the case of binary classification (AS = Binary) positives are assumed to
 * have labels equal to 1, whereas negatives are assumed to have labels equal to
 * 0.
 *
 * @tparam AS An average strategy.
 */
template<AverageStrategy AS>
class Recall
{
 public:
  /**
   * Run classification and calculate recall.
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
#include "recall_impl.hpp"

#endif
