/**
 * @file core/cv/metrics/f1.hpp
 * @author Kirill Mishchenko
 *
 * The F1 metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_F1_HPP
#define MLPACK_CORE_CV_METRICS_F1_HPP

#include <type_traits>

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/average_strategy.hpp>

namespace mlpack {

/**
 * F1 is a metric of performance for classification algorithms that for binary
 * classification is equal to @f$ 2 * precision * recall / (precision + recall)
 * @f$. For multiclass classification the F1 metric can be used with the
 * following strategies for averaging.
 * 1. Micro. The result is calculated by the above formula, but microaveraged
 * precision and microaveraged recall are used.
 * 2. Macro. F1 is calculated for each class (with values used for calculation
 * of macroaveraged precision and macroaveraged recall), and then the F1 values
 * are averaged.
 *
 * In the case of multiclass classification it is assumed that there are
 * instances of every label from 0 to max(labels) among input data points.
 *
 * The returned value for F1 will be zero if both precision and recall turn out
 * to be zeros.
 *
 * @tparam AS An average strategy.
 * @tparam PositiveClass In the case of binary classification (AS = Binary)
 *     positives are assumed to have labels equal to this value.
 */
template<AverageStrategy AS, size_t PositiveClass = 1>
class F1
{
 public:
  /**
   * Run classification and calculate F1.
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
   * Run classification and calculate F1 for binary classification.
   */
  template<AverageStrategy _AS,
           typename MLAlgorithm,
           typename DataType,
           typename = std::enable_if_t<_AS == Binary>>
  static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t>& labels);

  /**
   * Run classification and calculate microaveraged F1.
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
   * Run classification and calculate macroaveraged F1.
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
#include "f1_impl.hpp"

#endif
