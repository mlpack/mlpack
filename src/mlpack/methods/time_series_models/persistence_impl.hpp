/**
 * @file methods/time_series_models/persistence_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of the Persistence Model for time series data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_PERSISTENCE_MODEL_IMPL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_PERSISTENCE_MODEL_IMPL_HPP

// In case it hasn't yet been included.
#include "persistence.hpp"
#include "util/lag.hpp"

namespace mlpack {
namespace ts /* Time Series methods. */ {

PersistenceModel::PersistenceModel()
{
  // Nothing to do here.
}

template<typename InputType,
         typename = std::enable_if_t<arma::is_Row<InputType>::value ||
             arma::is_Col<InputType>::value>>
void PersistenceModel::Predict(const InputType& input,
    InputType& predictions)
{
    predictions = Lag(input, 1);
}

template<typename InputType,
         typename = std::enable_if_t<arma::is_Row<InputType>::value ||
             arma::is_Col<InputType>::value>>
void PersistenceModel::Predict(const InputType& train, const InputType& test,
             InputType& predictions)
{
    // Concatenating train and test.
    InputType concat(train.n_elem + test.n_elem);
    for(arma::uword i = 0; i < train.n_elem; i++)
    {
        concat(i) = train(i);
    }
    for(arma::uword i = 0; i < test.n_elem; i++)
    {
        concat(i + train.n_elem) = test(i);
    }
    
    InputType concat_pred(concat.n_elem);
    Predict(concat, concat_pred);

    // Taking the test predictions from all the predictions.
    predictions = concat_pred.tail(test.n_elem);
}

void PersistenceModel::Predict(const arma::mat& input,
    arma::rowvec& predictions)
{
    // Extracting the last column from the input as arma::rowvec.
    arma::colvec temp = input.col(input.n_cols - 1);
    arma::rowvec temp1 = temp.t();

    predictions = Lag(temp1, 1);
}

void PersistenceModel::Predict(const arma::mat& train, const arma::mat& test,
             arma::rowvec& predictions)
{
    // Concatenating train and test.
    arma::mat concat = arma::join_vert(train, test);

    arma::rowvec concat_pred(concat.n_rows);
    Predict(concat, concat_pred);

    // Taking the test predictions from all the predictions.
    predictions = concat_pred.tail(test.n_rows);
}

} // namespace ts
} // namespace mlpack

#endif
