/**
* @file topk_accuracy.hpp
* @author Arunav Shandeelya
*
* The TopK Accuracy [target is in topk prediction] Classification metric.

* mlpack is free software; you may redistribute it and/or modify it under the
* terms of the 3-clause BSD license.  You should have received a copy of the
* 3-clause BSD license along with mlpack.  If not, see
* http://www.opensource.org/licenses/BSD-3-Clause for more information.
*/

#ifndef MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_HPP
#define MLPACK_CORE_CV_METRICS_TOPK_ACCURACY_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv{

/**
 * The topk accuracy is a metric for classification algorithms
 * which gives top 'K' highest probability of the model. This metric
 * computes the number of times where the correct label is among the 
 * top 'K' label predicted (rank by predicted score).   
*/

class TopK_Accuracy
{
    public:
    /**
     * Run Prediction and calculate the topk accuracy score.
     * 
     * @param model A classification model
     * @param data Column-major containing test-items.
     * @params labels (Ground truth), target values for the test_items 
     * @params K, the top k label predicted class.
    */
   template<typename MLAlgorithm, typename DataType, typename ResponseType, typename TopK_Score>
   static double Evaluate(MLAlgorithm& model,
                        const DataType& data,
                        const ResponseType& labels,
                        const TopK_Score& k);
    /**
    * normalization : bool, default= True
    * If `True`, return the fraction of correctly classified samples.
    * Otherwise, return the number of correctly classified samples.
    */
    static const bool Normalization = true;
};

} // namespace cv 
} // namespace mlpack

#include "topk_accuracy_impl.hpp"
#endif


