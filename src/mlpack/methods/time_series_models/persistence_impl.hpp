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

template<typename InputType> 
void PersistenceModel::Predict(const InputType& input,
    arma::rowvec& predictions)
{
    predictions = Lag(input, 1);
}

void PersistenceModel::Predict(const arma::mat& input,
    arma::rowvec& predictions)
{
    // Extracting the last column from the input as arma::rowvec.
    arma::colvec temp = input.col(input.n_cols - 1);
    arma::rowvec temp1 = temp.t();
    
    predictions = Lag(temp1, 1);
}

} // namespace ts
} // namespace mlpack

#endif
