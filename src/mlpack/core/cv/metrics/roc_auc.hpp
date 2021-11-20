/**
 * @file core/cv/metrics/roc_auc.hpp
 * @author Suvarsha Chennareddy
 *
 * The Area Under the Receiver Operating Characteristic Curve (ROC AUC)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_ROC_AUC_HPP
#define MLPACK_CORE_CV_METRICS_ROC_AUC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * The ROC_AUC is a metric of performance for a binary classifiers (and even 
 * for multiclass classifiers) that measures the two-dimensional area 
 * underneath the entire ROC curve. For multiclass classifiers, a generalization 
 * implemented by Hand and Till is used.
 *
 */
template<size_t classification = 1>
class ROC_AUC
{
public:
  /**
   * Run prediction and calculate the ROC AUC.
   *
   * @param model A classifier that returns probabilities for 
   * @param data Column-major data containing test items.
   * @param responses Ground truth (correct) target values for the test items,
   *     should be either a row vector.
   */
template<typename MLAlgorithm, typename DataType, typename ResponsesType>
static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& responses);

  /**
   * Information for hyper-parameter tuning code. It indicates that we want
   * to minimize the measurement.
   */
static const bool NeedsMinimization = false;

private:
 /**
     Run classification and calculate ROC AUC for binary classification
 */
template<size_t _classification,
         typename MLAlgorithm,
         typename DataType,
         typename ResponsesType,
         typename = std::enable_if_t<_classification == 1>>
 static double Evaluate(MLAlgorithm& model,
            const DataType& data,
            const ResponsesType& responses);

/**
    Run classification and calculate AUC (generalization of the AUC established 
    by Hand and Till) for multiclass classification. 
*/
template<size_t _classification,
         typename MLAlgorithm,
         typename DataType,
         typename ResponsesType,
         typename = std::enable_if_t<_classification == 2>,
         typename = void>
static double Evaluate(MLAlgorithm& model,
        const DataType& data,
        const ResponsesType& responses);


template<typename ResponsesType>
static void sortRanks(arma::mat predictions, ResponsesType labels, size_t  i, size_t j, 
    size_t ni, size_t nj, arma::uvec& sortedRanks);

static size_t sumOfRanks(arma::uvec sortedRanks, size_t n);

};

} // namespace cv
} // namespace mlpack

// Include implementation.
#include "roc_auc_impl.hpp"

#endif
