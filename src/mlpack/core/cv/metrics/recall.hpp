/**
 * @file core/cv/metrics/recall.hpp
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

/**
 * Recall is a metric of performance for classification algorithms that for
 * binary classification is equal to @f$ tp / (tp + fn) @f$, where @f$ tp @f$
 * and @f$ fn @f$ are the numbers of true positives and false negatives
 * respectively. For multiclass classification the recall metric can be used
 * with the following strategies for averaging.
 * 1. Micro. If there are @f$ N + 1 @f$ classes in total, the result is equal to
 * @f[
 * (tp_0 + tp_1 + \ldots + tp_N) / (tp_0 + tp_1 + \ldots + tp_N + fn_0 + fn_1 +
 * \ldots + fn_N),
 * @f]
 * where @f$ tp_i @f$ and @f$ fn_i @f$ are the numbers of true
 * positives and false negatives respectively for the class (label) @f$ i @f$.
 * 2. Macro. If there are @f$ N + 1 @f$ classes in total, the result is equal to
 * the mean of the values
 * @f[
 * tp_0 / (tp_0 + fn_0), tp_1 / (tp_1 + fn_1), \ldots, tp_N / (tp_N + fn_N),
 * @f]
 * where @f$ tp_i @f$ and @f$ fn_i @f$ are the
 * numbers of true positives and false negatives respectively for the class
 * (label) @f$ i @f$.
 *
 * @tparam AS An average strategy.
 * @tparam PositiveClass In the case of binary classification (AS = Binary)
 *     positives are assumed to have labels equal to this value.
 */
template<AverageStrategy AS, size_t PositiveClass = 1>
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

 private:
  /**
   * Run classification and calculate recall for binary classification.
   */
  template<AverageStrategy _AS,
           typename MLAlgorithm,
           typename DataType,
           typename = std::enable_if_t<_AS == Binary>>
  static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t>& labels);

  /**
   * Run classification and calculate microaveraged recall.
   */
  template<AverageStrategy _AS,
           typename MLAlgorithm,
           typename DataType,
           typename = std::enable_if_t<_AS == Micro>,
           typename = void>
  static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t>& labels);

  /**
   * Run classification and calculate macroaveraged recall.
   */
  template<AverageStrategy _AS,
           typename MLAlgorithm,
           typename DataType,
           typename = std::enable_if_t<_AS == Macro>,
           typename = void,
           typename = void>
  static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t>& labels);
};

} // namespace mlpack

// Include implementation.
#include "recall_impl.hpp"

#endif
