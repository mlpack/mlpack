/**
* @file top_k_accuracy.hpp
* @author Arunav Shandeelya
*
* Definition of Top K Accuracy classification metric.

* mlpack is free software; you may redistribute it and/or modify it under the
* terms of the 3-clause BSD license.  You should have received a copy of the
* 3-clause BSD license along with mlpack.  If not, see
* http://www.opensource.org/licenses/BSD-3-Clause for more information.
*/

#ifndef MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_HPP
#define MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
 * The Top K Accuracy is a metric for classification algorithms
 * which gives top 'K' highest probability of the model. This metric
 * computes the number of times where the correct label is among the 
 * top 'K' label predicted (rank by predicted score).   
*/
class TopKAccuracy
{
 public:
  /**
   * Run Prediction and calculate the topk accuracy score.
   * 
   * @param model A classification model
   * @param data Column-major containing test-items.
   * @params labels (Ground truth), target values for the test_items 
   * @params k, the top k label predicted class.
   */
  template<typename MLAlgorithm, typename DataType, typename TopK>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const arma::Row<size_t>& labels,
                         const TopK& k);
  /**
   * normalization : bool, default= True
   * If `True`, return the fraction of correctly classified samples.
   * Otherwise, return the number of correctly classified samples.
   */
  static const bool Normalization = true;
};

} // namespace cv
} // namespace mlpack
// Include implementation.
#include "top_k_accuracy_impl.hpp"

#endif
