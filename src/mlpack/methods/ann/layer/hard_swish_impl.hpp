/**
 * @file methods/ann/layer/hard_swish_impl.hpp
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
#ifndef MLPACK_METHODS_ANN_LAYER_HARD_SWISH_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HARD_SWISH_IMPL_HPP

// In case it hasn't yet been included.
#include "hard_swish.hpp"

namespace mlpack {

    template<typename MatType>
    HardSwishType<MatType>::HardSwishType(const HardSwishType& other) :
        Layer<MatType>(other)
    {
        // Nothing to do.
    }

    template<typename MatType>
    HardSwishType<MatType>::HardSwishType(
        HardSwishType&& other) :
        Layer<MatType>(std::move(other))
    {
        // Nothing to do.
    }

    template<typename MatType>
    HardSwishType<MatType>&
        HardSwishType<MatType>::operator=(const HardSwishType& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(other);
        }

        return *this;
    }

    template<typename MatType>
    HardSwishType<MatType>&
        HardSwishType<MatType>::operator=(HardSwishType&& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(std::move(other));
        }

        return *this;
    }

    template<typename MatType>
    void HardSwishType<MatType>::Forward(
        const MatType& input, MatType& output)
    {

        output = (input >= 3) % (input) +
                (input < 3) % (input > -3) % (input % (input + 3.0)) / 6.0;

        if (this->training)
        {
            derivative.set_size(arma::size(input));
            derivative = (input >= 3) + 
                        (input < 3) % (input > -3) % ( 2 * input + 3.0) / 6.0;

        }
    }

    template<typename MatType>
    void HardSwishType<MatType>::Backward(
         const MatType&  input , const MatType& gy, MatType& g)
    {
        g = gy % derivative;
    }

    template<typename MatType>
    template<typename Archive>
    void HardSwishType<MatType>::serialize(
        Archive& ar,
        const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));

        if (Archive::is_loading::value)
            derivative.clear();
    }
}
#endif