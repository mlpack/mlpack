/**
 * @file mothods/time_series_models/persistence.hpp
 * @author Rishabh Garg
 *
 * Definition of the Persistence Model for time series data.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_PERSISTENCE_MODEL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_PERSISTENCE_MODEL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ts /* Time Series methods. */ {

/**
 * Persistence model is the simplest time series model. This is sometimes also
 * called as the naive forecasting method. It simply outputs the value of the
 * previous timestamp as the prediction for the current timestamp.
 */
class PersistenceModel
{
 public:
  /**
   * Creates a model.
   * 
   */
  PersistenceModel() {};
  
  template<typename InputType>
  void Predict(const InputType& input, arma::rowvec& predictions);
  
  /**
   * This speciaization takes as input an arma::mat and assumes that the column
   * required for the time series analysis is present as the last column.
   */
  void Predict(const arma::mat& input, arma::rowvec& predictions);
  
}; // class PersistenceModel




} // namespace ts
} // namespace mlpack

// Include implementation.
#include "persistence_impl.hpp"

#endif
