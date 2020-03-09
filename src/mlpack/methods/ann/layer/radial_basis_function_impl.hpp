/**
 * @file linear_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Linear layer class also known as fully-connected layer
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
#include "radial_basis_function.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
RBF<InputDataType, OutputDataType>::RBF() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
RBF<InputDataType, OutputDataType, RegularizerType>::RBF(
    const size_t inSize,
    const size_t outSize) :
    inSize(inSize),
    outSize(outSize)
{
 // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void RBF<InputDataType, OutputDataType, RegularizerType>::Reset()
{
  centres = arma::randu(outSize, inSize);
  centres = arma::normcdf(centres, 0, 1);
  sigmas = arma::ones(outSize);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void RBF<InputDataType, OutputDataType, RegularizerType>::Forward(
    const InputDataType&& input, OutputDataType&& output)
{  
  arma::cube x = arma::cube(input.n_rows, outSize, inSize);
  for(int i=0;i<inSize;i++)
  {
    for (int j = 0;j<target.n_cols; j++)
    {
      input.slice(i).row(j)= target.row(i);
    }
  }
  arma::cube c = arma::cube(input.n_rows, outSize, inSize);
  for(size_t i=0; i < inSize; i++)
  {
    input.slice(i)= centres;
  }
  distances = arma::pow (arma::sum (arma::pow ((x - c), 2), 1), 0.5) * sigmas;
  output = distances;

}
template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename InputType, typename ErrorType, typename GradientType>
void RBF<InputDataType, OutputDataType, RegularizerType>::Backward(
    const InputDataType&& /* input */, ErrorType&& gy, GradientType&& g)
{
  g = distances.t() * gy;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename Archive>
void RBF<InputDataType, OutputDataType, RegularizerType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);
}

} // namespace ann
} // namespace mlpack

#endif
