/**
 * @file methods/ann/layer/gelu_function.hpp
 * @author Vivek Pal
 * @author Dakshit Agrawal
 *
 * Definition and implementation of the Gaussian Error Linear Unit (GELU)
 * function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GELU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GELU_IMPL_HPP

// In case it hasn't yet been included.
#include "gelu.hpp"

namespace mlpack {

    template<typename MatType>
    GELUType<MatType>::GELUType(const GELUType& other) :
        Layer<MatType>(other)
    {
        // Nothing to do.
    }

    template<typename MatType>
    GELUType<MatType>::GELUType(
        GELUType&& other) :
        Layer<MatType>(std::move(other))
    {
        // Nothing to do.
    }

    template<typename MatType>
    GELUType<MatType>&
        GELUType<MatType>::operator=(const GELUType& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(other);
        }

        return *this;
    }

    template<typename MatType>
    GELUType<MatType>&
        GELUType<MatType>::operator=(GELUType&& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(std::move(other));
        }

        return *this;
    }

    template<typename MatType>
    void GELUType<MatType>::Forward(
        const MatType& input, MatType& output)
    {

        output = 0.5 * input % (1 + arma::tanh(std::sqrt(2 / M_PI) *
        (input + 0.044715 * arma::pow(input, 3))));

        if (this->training)
        {
            derivative.set_size(arma::size(input));
            derivative = 0.5 * arma::tanh(0.0356774 * arma::pow(input, 3) + 
            0.797885 * input) + (0.0535161 * arma::pow(input, 3) + 0.398942 * 
            input) % arma::pow(1 / arma::cosh(0.0356774 * arma::pow(input, 3) +
            0.797885 * input), 2) + 0.5;

        }
    }

    template<typename MatType>
    void GELUType<MatType>::Backward(
         const MatType&  input , const MatType& gy, MatType& g)
    {
        g = gy % derivative;
    }

    template<typename MatType>
    template<typename Archive>
    void GELUType<MatType>::serialize(
        Archive& ar,
        const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));

        if (Archive::is_loading::value)
            derivative.clear();
    }
}
#endif