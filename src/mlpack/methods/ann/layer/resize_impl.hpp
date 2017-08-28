/**
 * @file resize_impl.hpp
 * @author Kris Singh
 *
 * Implementation of the Resize layer class also known as fully-connected layer
 * or affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RESIZE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RESIZE_IMPL_HPP

// In case it hasn't yet been included.
#include <mlpack/methods/ann/image_functions/bilinear_function.hpp>
#include "resize.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<class InterpolationType,
         typename InputDataType,
         typename OutputDataType
         >
Resize<InterpolationType, InputDataType, OutputDataType>::
Resize(InterpolationType policy): policy(policy)
{
  // Nothing to do here.
}

template<class InterpolationType,
         typename InputDataType,
         typename OutputDataType
         >
template<typename eT>
void Resize<InterpolationType, InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  policy.UpSample(input, output);
}

template<class InterpolationType,
         typename InputDataType,
         typename OutputDataType
         >
template<typename eT>
void Resize<InterpolationType, InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  policy.DownSample(input, gy, g);
}

template<class InterpolationType,
         typename InputDataType,
         typename OutputDataType
         >
template<typename Archive>
void Resize<InterpolationType, InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(policy, "policy");
}

} // namespace ann
} // namespace mlpack

#endif
