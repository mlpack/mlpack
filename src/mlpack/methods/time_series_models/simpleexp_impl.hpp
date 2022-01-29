/**
 * @file methods/time_series_models/simpleexpo_impl.hpp
 * @author Rishabh Bali
 *
 * Implementation of the SimpleExpo Model for time series data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_IMPL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_IMPL_HPP

#include <mlpack/prereqs.hpp>

// In case it hasn't been included already

#include "simpleexpo.hpp"

namespace mlpack
{
    namespace ts
    {
        SimpleExpo::SimpleExpo()
        { /* Nothing to do here */}
        

        SimpleExpo::SimpleExpo(const arma::rowvec &data, const double alpha) : alpha(alpha),datapts(data)
        {
            level = 0;
        }

        SimpleExpo::SimpleExpo(const arma::rowvec &data) : datapts(data)
        {
            alpha = ((double)rand() / (RAND_MAX));
        }

        SimpleExpo::SimpleExpo(const arma::mat &data, const double &alpha) : alpha(alpha)
        {
            datapts = data.row(data.n_rows - 1);
        }

        SimpleExpo::SimpleExpo(const arma::mat &data)
        {
            alpha = ((double)rand() / (RAND_MAX));
            datapts = data.row(data.n_rows - 1);
        }
         double & SimpleExpo::Alpha() 
        {
            return alpha;
        }
        double & SimpleExpo::UpAlpha()
        {
            // return updated alpha 
        }

    } //namespace ts
} // namespace mlpack

#endif
