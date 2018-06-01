/**
 * @file sampling_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Sampling class which samples from parameters for a given
 * distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SAMPLING_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_SAMPLING_IMPL_HPP

// In case it hasn't yet been included.
#include "sampling.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Sampling<InputDataType, OutputDataType>::Sampling()
{
  // Nothing to do here.
}

// template <typename InputDataType, typename OutputDataType>
// Sampling<InputDataType, OutputDataType>::Sampling(
//     const size_t inSize,
//     const size_t outSize) :
//     inSize(inSize),
//     outSize(outSize)
// {
//   weights.set_size(2 * outSize * inSize + 2 * outSize, 1);
// }

template <typename InputDataType, typename OutputDataType>
Sampling<InputDataType, OutputDataType>::Sampling(
    const size_t sampleSize) :
    outSize(sampleSize)
{
  // Nothing to do here.
}

// template<typename InputDataType, typename OutputDataType>
// void Sampling<InputDataType, OutputDataType>::Reset()
// {
//   weights.set_size(2 * outSize * inSize + 2 * outSize, 1);

//   weight = arma::mat(weights.memptr(), 2 * outSize, inSize, false, false);
//   bias = arma::mat(weights.memptr() + weight.n_elem,
//       2 * outSize, 1, false, false);
// }

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sampling<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  arma::arma_rng::set_seed_random();

  // output = weight * input;
  // output.each_col() += bias;
  gaussianSample = arma::randn<arma::mat>(outSize, input.n_cols);
  output = (input.submat(outSize, 0, 2 * outSize - 1, input.n_cols - 1) +
      input.submat(0, 0, outSize - 1, input.n_cols - 1)) % gaussianSample;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sampling<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{std::cout << gy.n_rows << std::endl << gy.n_cols << std::endl;
  g = join_cols((weight.submat(0, 0, outSize - 1, inSize - 1).t() * gy) % 
      gaussianSample, 
      weight.submat(outSize, 0, 2 * outSize - 1, inSize - 1).t() * gy);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Sampling<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      error * input.t());
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(error, 1);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Sampling<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(2 * outSize * inSize + 2 * outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif