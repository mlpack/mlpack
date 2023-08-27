/**
 * @file methods/ann/activation_functions/elish_function.hpp
 * @author Bisakh Mondal
 *
 * Definition and implementation of the ELiSH function as described by
 * Mina Basirat and Peter M. Roth.
 *
 * For more information see the following paper
 *
 * @code
 * @misc{Basirat2018,
 *    title = {The Quest for the Golden Activation Function},
 *    author = {Mina Basirat and Peter M. Roth},
 *    year = {2018},
 *    url = {https://arxiv.org/pdf/1808.00783.pdf},
 *    eprint = {1808.00783},
 *    archivePrefix = {arXiv},
 *    primaryClass = {cs.NE} }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELISH_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ELISH_FUNCTION_IMPL_HPP

// In case it hasn't yet been included.
#include "elish_function.hpp"

namespace mlpack {

    template<typename MatType>
    ElishType<MatType>::ElishType(const ElishType& other) :
        Layer<MatType>(other)
    {
        // Nothing to do.
    }

    template<typename MatType>
    ElishType<MatType>::ElishType(
        ElishType&& other) :
        Layer<MatType>(std::move(other))
    {
        // Nothing to do.
    }

    template<typename MatType>
    ElishType<MatType>&
        ElishType<MatType>::operator=(const ElishType& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(other);
        }

        return *this;
    }

    template<typename MatType>
    ElishType<MatType>&
        ElishType<MatType>::operator=(ElishType&& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(std::move(other));
        }

        return *this;
    }

    template<typename MatType>
    void ElishType<MatType>::Forward(
        const MatType& input, MatType& output)
    {

        output = ((input < 0.0) % ((arma::exp(input) - 1) / (1 + arma::exp(-input))))
            + ((input >= 0.0) % (input / (1 + arma::exp(-input))));

        if (this->training)
        {
            derivative.set_size(arma::size(input));
            derivative = ((input < 0.0) % (arma::exp(input) - 2 / (1 + arma::exp(input)) + 2 / arma::pow(
                1 + arma::exp(input), 2))) + ((input >= 0.0) % (1 / (1 + arma::exp(-input)) + input %
                    arma::exp(-input) / arma::pow(1 + arma::exp(-input), 2)));

        }
    }

    template<typename MatType>
    void ElishType<MatType>::Backward(
         const MatType&  input , const MatType& gy, MatType& g)
    {
        g = gy % derivative;
    }

    template<typename MatType>
    template<typename Archive>
    void ElishType<MatType>::serialize(
        Archive& ar,
        const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));

        if (Archive::is_loading::value)
            derivative.clear();
    }
}
#endif