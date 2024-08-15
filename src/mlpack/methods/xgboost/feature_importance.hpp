/*
 * @file methods/xgboost/feature_importance.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the Feature Importance class in xgboost.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_FEATURE_IMPORTANCE_HPP
#define MLPACK_METHODS_XGBOOST_FEATURE_IMPORTANCE_HPP

#include "xgboost.hpp"
#include <mlpack/core.hpp>

using namespace std;

namespace mlpack {

/**
 * After training, XGBoost can calculate feature importance to understand 
 * the contribution of each feature in the classification decision.
 * XGBoost provides two main types of feature importance scores:
 * 
 * - Weight (Frequency): This is the number of times a feature is used to 
 *   split a node across all trees in the model. It counts how often each 
 *   feature appears in all trees of the model.
 * 
 * - Gain (Cover): This measures the improvement in accuracy brought by a 
 *   feature to the model. For each feature, it sums the improvement in accuracy 
 *   (reduction in loss) brought by the feature when it is used in tree splits.
 */

class FeatureImportance {

  public:
   
  map<size_t, size_t> featureFrequency;
  map<size_t, double> featureCover;
  vector<size_t> rankByFrequency;
  vector<size_t> rankByCover;

  // Empty class constructor.
  FeatureImportance() { /*Nothing to do*/ }

  //! Edit the featureFrequency value.
  void increaseFeatureFrequency(size_t index, size_t incrementValue)
  {
    featureFrequency[index] += incrementValue;
  }

  //! Edit the featureCover value.
  void increaseFeatureCover(size_t index, double incrementValue)
  {
    featureCover[index] += incrementValue;
  }

  void RankByFrequency()
  {
    priority_queue<pair<size_t, size_t>> pq; 

    map<size_t, size_t>::iterator it = featureFrequency.begin(); 

    for(; it != featureFrequency.end(); ++it)
      pq.push({-it->second, it->first});

    while(!pq.empty())
    {
      size_t elem = pq.top().second; 
      pq.pop();
      rankByFrequency.push_back(elem);
    }
  }

  void RankByCover()
  {
    priority_queue<pair<double, size_t>> pq; 

    map<size_t, double>::iterator it = featureCover.begin(); 

    for(; it != featureCover.end(); ++it)
      pq.push({-(it->second), it->first});

    while(!pq.empty())
    {
      size_t elem = pq.top().second; 
      pq.pop();
      rankByCover.push_back(elem);
    }
  }

};

}

#endif