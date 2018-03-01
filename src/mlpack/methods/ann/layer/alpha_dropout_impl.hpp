/**
 * @file alpha_dropout_impl.hpp
 * @author Dakshit Agrawal
 *
 * Definition of the Alpha-Dropout class, which implements a regularizer that
 * randomly sets units to alpha-dash to prevent them from co-adapting and
 * makes an affine transformation so as to keep the mean and variance of
 * outputs at their original values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ALPHA_DROPOUT_IMPL_HPP

// In case it hasn't yet been included.
#include "alpha_dropout.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Alpha_Dropout<InputDataType, OutputDataType>::Alpha_Dropout(
        const double ratio) :
        ratio(ratio),
        deterministic(true)
{
    // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Alpha_Dropout<InputDataType, OutputDataType>::Forward(
        const arma::Mat<eT>&& input,
        arma::Mat<eT>&& output)
{
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
       output = input;
    }
    else
    {
        // Set values to alpha_dash with probability ratio.  Then apply affine
        // transformation so as to keep mean and variance of outputs to their
        // original values.

        mask = arma::randu< arma::Mat<eT> >(input.n_rows, input.n_cols);
        mask.transform( [&](double val) { return (val > ratio); } );
        output = input % mask + alpha_dash * (1 - mask);

        output = (output * a) + b;
    }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Alpha_Dropout<InputDataType, OutputDataType>::Backward(
        const arma::Mat<eT>&& /* input */,
        arma::Mat<eT>&& gy,
        arma::Mat<eT>&& g)
{
    g = gy % mask * a;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Alpha_Dropout<InputDataType, OutputDataType>::serialize(
        Archive& ar,
        const unsigned int /* version */)
{
    ar & BOOST_SERIALIZATION_NVP(ratio);

    // Reset values of a and b
    a = pow((1 - ratio) * (1 + ratio * pow(alpha_dash, 2)), -0.5);

    b = -a * alpha_dash * ratio;
}

} // namespace ann
} // namespace mlpack

#endif
