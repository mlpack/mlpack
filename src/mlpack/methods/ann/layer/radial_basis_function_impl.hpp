/**
 * @file radial_basis_impl.hpp
 * @author Himanshu Pathak
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_RBF_IMPL_HPP

// In case it hasn't yet been included.
#include "dropout.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
RBF<InputDataType, OutputDataType, RegularizerType>::RBF() :
    inSize(0),
    outSize(0),
    reset(false)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
RBF<InputDataType, OutputDataType, RegularizerType>::RBF(
    const size_t inSize,
    const size_t outSize,
    arma::mat& centres,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    centres(centres),
    regularizer(regularizer),
    reset(false)
{
  weights.set_size(outSize * inSize + outSize, 1);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void RBF<InputDataType, OutputDataType, RegularizerType>::Reset()
{
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void RBF<InputDataType, OutputDataType, RegularizerType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  if(!reset)
  {
    arma::mat sigmas = arma::mat(1, outSize);
    sigmas.ones();
    sigmas = sigmas / outSize;
    reset = true;
  }
  distances = arma::mat(outSize, input.n_cols);

  for (size_t i = 0; i < input.n_cols; i++)
  {
    arma::mat temp = centres.each_col() - input.col(i);
    distances.col(i) = arma::pow(arma::sum(
                                 arma::pow((temp),
                                 2), 0), 0.5).t();
  }

  sigmas = arma::mean(distances, 1);
  arma::mat betas = 1 / 2 * arma::pow(sigmas, 2);
  distances = arma::pow(distances, 2);
  distances = distances.each_col() % betas;
  output = arma::exp(-1 * distances);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void RBF<InputDataType, OutputDataType, RegularizerType>::Backward(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  g = centres.t() * gy;
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename Archive>
void RBF<InputDataType, OutputDataType, RegularizerType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(distances);
  ar & BOOST_SERIALIZATION_NVP(sigmas);
  ar & BOOST_SERIALIZATION_NVP(centres);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(outSize * inSize + outSize, 1);
}

} // namespace ann
} // namespace mlpack
#endif
