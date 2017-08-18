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
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_IMPL_HPP

// In case it hasn't yet been included.
#include <mlpack/methods/ann/image_functions/bilinear_function.hpp>
#include "resize.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType,
         typename OutputDataType,
         typename InterpolationType>
Resize<InputDataType, OutputDataType, InterpolationType>::
Resize(InterpolationType policy): policy(policy)
{
  // Nothing to do here.
}

template<typename InputDataType,
         typename OutputDataType,
         typename InterpolationType>
template<typename eT>
void Resize<InputDataType, OutputDataType, InterpolationType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  policy.UpSample(input, output);
}

template<typename InputDataType,
         typename OutputDataType,
         typename InterpolationType>
template<typename eT>
void Resize<InputDataType, OutputDataType, InterpolationType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{

  policy.DownSample(gy, g);
  g.resize(g.n_rows * g.n_cols , 1);
}

template<typename InputDataType,
         typename OutputDataType,
         typename InterpolationType>
template<typename Archive>
void Resize<InputDataType, OutputDataType, InterpolationType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(policy, "policy");
}

} // namespace ann
} // namespace mlpack

#endif
