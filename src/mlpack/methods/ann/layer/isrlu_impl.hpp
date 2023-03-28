/**
 * @file methods/ann/layer/isrlu_impl.hpp
 * @author Abhinav Anand
 *
 * Implementation of the ISRLU activation function as described by Jonathan T. Barron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ISRLU_IMPL_HPP

// In case it hasn't yet been included.
#include "isrlu.hpp"
#include <omp.h>

namespace mlpack {
    template<typename MatType>
    ISRLU<MatType>::ISRLU(
        const ISRLU& other) :
        Layer<MatType>(other),
        alpha(other.alpha)
       
    {
        // Nothing to do here.
    }

    template<typename MatType>
    ISRLU<MatType>::ISRLU(
        ISRLU&& other) :
        Layer<MatType>(std::move(other)),
        alpha(std::move(other.alpha))
    {
        // Nothing to do here.
    }

    template<typename MatType>
    ISRLU<MatType>&
        ISRLU<MatType>::operator=(const ISRLU& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(other);
            alpha = other.alpha;
        }

        return *this;
    }

    template<typename MatType>
    ISRLU<MatType>&
        ISRLU<MatType>::operator=(ISRLU&& other)
    {
        if (&other != this)
        {
            Layer<MatType>::operator=(std::move(other));
            alpha = other.alpha;
        }

        return *this;
    }

template<typename MatType>
ISRLU<MatType>::ISRLU(const double alpha) :
    Layer<MatType>(),
    alpha(alpha)
{
    //Nothing to do here
}


template<typename MatType>
void ISRLU<MatType>::Forward(
    const MatType& input, MatType& output)
{
    output.ones(arma::size(input));
  
  #pragma omp parallel for  
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output(i) = (input(i) >= 0) ? input(i) : input(i) *
        (1 / std::sqrt(1 + alpha * (input(i) * input(i))));
  }
}


template<typename MatType>
void ISRLU<MatType>::Backward(
    const MatType& input, const MatType& gy, MatType& g)
{
  derivative.set_size(arma::size(input));
  
  #pragma omp parallel for  
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    derivative(i) = (input(i) >= 0) ? 1 :
        std::pow(1 / std::sqrt(1 + alpha * input(i) * input(i)), 3);
  }
  g = gy % derivative;
}

template<typename MatType>
template<typename Archive>
void ISRLU<MatType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(alpha));
}

} // namespace mlpack

#endif