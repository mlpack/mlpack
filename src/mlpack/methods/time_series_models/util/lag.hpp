/**
 * @file mothods/time_series_models/util/lag.hpp
 * @author Rishabh Garg
 *
 * Implementation of Lag() which returns the feature shifted by a specified
 * period.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_UTIL_LAG_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_UTIL_LAG_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>

namespace mlpack {
namespace ts /* Time Series methods. */ {

/**
 * It returns the lagged feature of the input time series vector.
 *
 * @tparam InputType Type of input (arma::rowvec or arma::colvec).
 * @param input Input data to create lagged feature.
 * @param k Number of periods to shift (must be a non negative integer).
 */
template<typename InputType>
InputType Lag(const InputType& input, size_t k) {
    InputType lag_k(input.n_elem);

    if(k == input.n_elem) {
        Log::Warn << "The value of k is equal to the number of elements in the"
        << " input vector. Output vector will contain all NaNs" << std::endl;
    }

    if(k > input.n_elem) {
        Log::Fatal << "The value of k i.e. " << k << " must be less than or"
        << " equal to the number of elements of input vector i.e. "
        << input.n_elem << "." << std::endl;
    }

    // t = 0 to k doesn't have data corresponding to t-k timestamp. So, they
    // are initialized as NaNs.
    for(arma::uword i = 0; i < k; i++) {
        lag_k(i) = arma::datum::nan;
    }

    for(arma::uword i = k; i < input.n_elem; i++) {
        lag_k(i) = input(i-k);
    }
    return lag_k;
}

} // namespace ts
} // namespace mlpack

#endif
