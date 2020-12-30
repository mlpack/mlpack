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
   * Creates the model.
   *
   */
  PersistenceModel();
  
  /**
   * This overload can be used to make predictions on train dataset.
   *
   * @tparam InputType Type of input (arma::Row or arma::Col).
   * @param input Input data.
   * @param predictions To store the predictions made on input data.
   */
  template<typename InputType,
           typename = std::enable_if_t<arma::is_Row<InputType>::value ||
               arma::is_Col<InputType>::value>>
  void Predict(const InputType& input, InputType& predictions);

  /**
   * This overload can be used to make predicions on test dataset.
   * It assumes that the first row of test data is the successor of the last
   * row of train data.
   *
   * @tparam InputType Type of input (arma::Row or arma::Col).
   * @param train Train data.
   * @param test Test data.
   * @param predictions To store the predictions made on test data.
   */
  template<typename InputType,
           typename = std::enable_if_t<arma::is_Row<InputType>::value ||
               arma::is_Col<InputType>::value>>
  void Predict(const InputType& train, const InputType& test,
               InputType& predictions);

  /**
   * This overload takes as input an arma::mat and assumes that the column
   * required for the time series analysis is present as the last column.
   * It can be used to make predictions on train dataset.
   *
   * @param input Input data.
   * @param predictions To store the predictions made on input data.
   */
  void Predict(const arma::mat& input, arma::rowvec& predictions);

  /**
   * This overload can be used to make predicions on test dataset.
   * It assumes that the first row of test data is the successor of the last
   * row of train data.
   * This also assumes that the column required for the time series analysis
   * is present as the last column.
   *
   * @param train Train data.
   * @param test Test data.
   * @param presictions To store the predictions made on test data.
   */
  void Predict(const arma::mat& train, const arma::mat& test,
               arma::rowvec& predictions);

}; // class PersistenceModel

} // namespace ts
} // namespace mlpack

// Include implementation.
#include "persistence_impl.hpp"

#endif
