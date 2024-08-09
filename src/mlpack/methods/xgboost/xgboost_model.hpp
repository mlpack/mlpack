/**
 * @file methods/xgboost/xgboost_model.hpp
 * @author Abhimanyu Dayal
 *
 * XGBoost model used by the XGBoost binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_MODEL_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_MODEL_HPP

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Use forward declaration instead of include to accelerate compilation.
class XGBoost; 

// Defining the XGBoostModel class within the mlpack namespace.
namespace mlpack {

/**
 * The model to save to disk.
 */
class XGBoostModel 
{

};

}

// Include implementation.
#include "xgboost_model_impl.hpp"

#endif