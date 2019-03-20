/**
 * @file specificity.hpp
 * @author Gaurav Sharma
 *
 * The specificity metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_SPECIFICITY_HPP
#define MLPACK_CORE_CV_METRICS_SPECIFICITY_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/cv/metrics/average_strategy.hpp>

namespace mlpack {
namespace cv {

/**
 * Specificity is a metric of performance for classification algorithms
 * measures the proportion of negatives which are correctly identified
 * and also known as "true negative rate".
 * For binary classification is equal to @f$ tn / (tn + fp) @f$, where @f$ tn @f$
 * and @f$ fp @f$ are the numbers of true negatives and false positives
 * respectively. For multiclass classification the specificity metric can be used
 * with the following strategies for averaging.
 * 1. Micro. If there are @f$ N + 1 @f$ classes in total, the result is equal to
 * @f[
 * (tn_0 + tn_1 + \ldots + tn_N) / (tn_0 + tn_1 + \ldots + tn_N + fp_0 + fp_1 +
 * \ldots + fp_N),
 * @f]
 * where @f$ tn_i @f$ and @f$ fp_i @f$ are the numbers of true
 * negatives and false positives respectively for the class (label) @f$ i @f$.
 * 2. Macro. If there are @f$ N + 1 @f$ classes in total, the result is equal to
 * the mean of the values
 * @f[
 * tn_0 / (tn_0 + fp_0), tn_1 / (tn_1 + fp_1), \ldots, tn_N / (tn_N + fp_N),
 * @f]
 * where @f$ tn_i @f$ and @f$ fp_i @f$ are the
 * numbers of true negatives and false positives respectively for the class
 * (label) @f$ i @f$.
 *
 * @tparam AS An average strategy.
 * @tparam PositiveClass In the case of binary classification (AS = Binary)
 *     positives are assumed to have labels equal to this value.
 */
template<AverageStrategy AS, size_t PositiveClass = 1>
class Specificity
{
 public:
  /**
   * Run classification and calculate specificity.
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
   * Run classification and calculate specificity for binary classification.
   */
  template<AverageStrategy _AS,
           typename MLAlgorithm,
           typename DataType,
           typename = std::enable_if_t<_AS == Binary>>
  static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const arma::Row<size_t>& labels);

  /**
   * Run classification and calculate microaveraged specificity.
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
   * Run classification and calculate macroaveraged specificity.
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

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "specificity_impl.hpp"

#endif
