/**
 * @file methods/xgboost/xgboost.hpp
 * @author Abhimanyu Dayal
 *
 * XGBoost class. XGBoost optimises the Gradient Boosting algorithm by implementing
 * various addition functionalities on top of it such as regularisation, pruning etc.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_HPP

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Written in mlpack namespace.
namespace mlpack {

template<typename WeakLearnerType = XGBTree, typename MatType = arma::mat>
class XGBoost
{
  XGBoost() {/*Nothing to do*/}

  private:

  template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
  void TrainInternal(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numModels,
                const size_t numClasses,
                const WeakLearnerType& wl,
                WeakLearnerArgs&&... weakLearnerArgs);
};

}

// Include implementation.
#include "xgboost_impl.hpp"


#endif