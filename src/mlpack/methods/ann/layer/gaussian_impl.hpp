/**
 * @file gaussian_function.hpp
 * @author Himanshu Pathak
 *
 * Definition and implementation of the gaussian function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GAUSSIAN_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_GAUSSIAN_IMPL_HPP

// In case it hasn't yet been included.
#include "gaussian.hpp"

namespace mlpack {

    template<typename MatType>
    GaussianType<MatType>::GaussianType(const GaussianType& other) :
        Layer<MatType>(other)
    {
        // Nothing to do.
    }

    template<typename MatType>
    GaussianType<MatType>::GaussianType(
        GaussianType&& other) :
        Layer<MatType>(std::move(other))
    {
        // Nothing to do.
    }

    template<typename MatType>
    GaussianType<MatType>&
        GaussianType<MatType>::operator=(const GaussianType& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(other);
        }

        return *this;
    }

    template<typename MatType>
    GaussianType<MatType>&
        GaussianType<MatType>::operator=(GaussianType&& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(std::move(other));
        }

        return *this;
    }

    template<typename MatType>
    void GaussianType<MatType>::Forward(
        const MatType& input, MatType& output)
    {

        output = arma::exp(-1 * arma::pow(input, 2));

        if (this->training)
        {
            derivative.set_size(arma::size(input));
            derivative = 2 * -input % arma::exp(-1 * arma::pow(input, 2));

        }
    }

    template<typename MatType>
    void GaussianType<MatType>::Backward(
         const MatType&  input , const MatType& gy, MatType& g)
    {
        g = gy % derivative;
    }

    template<typename MatType>
    template<typename Archive>
    void GaussianType<MatType>::serialize(
        Archive& ar,
        const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));

        if (Archive::is_loading::value)
            derivative.clear();
    }
}
#endif