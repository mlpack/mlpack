/*
 * @file methods/xgboost/feature_importance.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the featureImportance class in xgboost.
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

namespace mlpack {

/**
 * After training, xgboost can calculate feature importance to understand 
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

}

#endif