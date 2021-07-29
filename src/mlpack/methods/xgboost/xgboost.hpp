/**
 * @file methods/xgboost/xgboost.hpp
 * @author Rishabh Garg
 *
 * Definition of the XGBoost class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_HPP

namespace mlpack {
namespace ensemble {

/**
 * The XGboost class provides the implementation of Gradient Boosting as
 * described in the XGBoost paper:
 *
 * @code
 * @inproceedings{Chen:2016:XST:2939672.2939785,
 *   author = {Chen, Tianqi and Guestrin, Carlos},
 *   title = {{XGBoost}: A Scalable Tree Boosting System},
 *   booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on
 *                Knowledge Discovery and Data Mining},
 *   series = {KDD '16},
 *   year = {2016},
 *   isbn = {978-1-4503-4232-2},
 *   location = {San Francisco, California, USA},
 *   pages = {785--794},
 *   numpages = {10},
 *   url = {http://doi.acm.org/10.1145/2939672.2939785},
 *   doi = {10.1145/2939672.2939785},
 *   acmid = {2939785},
 *   publisher = {ACM},
 *   address = {New York, NY, USA},
 *   keywords = {large-scale machine learning},
 * }
 * @endcode
 */
template<
    typename WeakLearnerType =
        mlpack::tree::DecisionTreeRegressor<
            SSELoss,
            XGBExactNumericSplit>
>
class XGBoost
{
 public:
  /**
   * Construct the xgboost without any training or specifying the number
   * of trees.  Predict() will throw an exception until Train() is called.
   */
  XGBoost();

  /**
   * Construct the xgboost forest
   */
  template<typename MatType, typename ResponsesType>
  XGBoost(const MatType& dataset,
          const ResponsesType& responses,
          const size_t numTrees = 100,
          const size_t maxDepth = 6,
          const double eta = 0.3,
          const double minChildWeight = 1,
          const double gamma = 0,
          const double lambda = 1,
          const double alpha = 0,
          const size_t numDimensions = 0);
}

} // namespace ensemble
} // namespace mlpack

// Include implementation.
#include "xgboost_impl.hpp"

#endif
