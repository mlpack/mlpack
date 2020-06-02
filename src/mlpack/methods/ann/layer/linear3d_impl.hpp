/**
 * @file methods/ann/layer/linear3d_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Linear layer class which accepts 3D input.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR3D_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR3D_IMPL_HPP

// In case it hasn't yet been included.
#include "linear3d.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear3D<InputDataType, OutputDataType, RegularizerType>::Linear() :
    inSize(0),
    outSize(0)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
Linear3D<InputDataType, OutputDataType, RegularizerType>::Linear(
    const size_t inSize,
    const size_t outSize,
    RegularizerType regularizer) :
    inSize(inSize),
    outSize(outSize),
    regularizer(regularizer)
{
  weights.set_size(outSize * inSize + outSize, 1);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
void Linear3D<InputDataType, OutputDataType, RegularizerType>::Reset()
{
  weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
  bias = arma::mat(weights.memptr() + weight.n_elem,
      outSize, 1, false, false);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void Linear3D<InputDataType, OutputDataType, RegularizerType>::Forward(
    const arma::Mat<eT>& input, arma::Mat<eT>& output)
{
  Log::Assert(input.n_rows % inSize == 0);
  output.set_size(outSize * input.n_rows / inSize, input.n_cols);
  arma::Cube<eT> inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(), inSize,
      input.n_rows / inSize, input.n_cols, false, false);

  #pragma omp parallel for
  for (omp_size_t i = 0; i < (omp_size_t) inputTemp.n_slices; ++i)
  {
    output.col(i) = arma::vectorise(weight * inputTemp.slice(i) +
        arma::repmat(bias, 1, input.n_rows / inSize));
  }
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void Linear3D<InputDataType, OutputDataType, RegularizerType>::Backward(
    const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
{
  Log::Assert(gy.n_rows % inSize == 0);
  g.set_size(outSize * gy.n_rows / inSize, batchSize);
  arma::Cube<eT> gyTemp(const_cast<arma::Mat<eT>&>(gy).memptr(),
      outSize, gy.n_rows / inSize, batchSize, false, false);

  #pragma omp parallel for
  for (omp_size_t i = 0; i < (omp_size_t) gyTemp.n_slices; ++i)
  {
    g.col(i) = arma::vectorise(weight.t() * gyTemp.slice(i));
  }
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename eT>
void Linear3D<InputDataType, OutputDataType, RegularizerType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  Log::Assert(input.n_cols == error.n_cols);
  Log::Assert(input.n_rows % inSize == 0);
  arma::Cube<eT> inputTemp(const_cast<arma::mat<eT>&>(input).memptr(), inSize,
      input.n_rows / inSize, input.n_cols, false, false);
  arma::Cube<eT> errorTemp(const_cast<arma::mat<eT>&>(error).memptr(), outSize,
      input.n_rows / inSize, error.n_cols, false, false);

  arma::Cube<eT> dW(outSize, inSize, input.n_cols);
  #pragma omp parallel for
  for (omp_size_t i = 0; i < (omp_size_t) input.n_cols; ++i)
  {
    dW.slice(i) = errorTemp.slice(i) * inputTemp.slice(i).t();
  }

  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      arma::sum(dW, 2) / inputTemp.n_slices);
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) =
      arma::sum(arma::sum(errorTemp, 2), 1) / erroTemp.n_slices;
  regularizer.Evaluate(weights, gradient);
}

template<typename InputDataType, typename OutputDataType,
    typename RegularizerType>
template<typename Archive>
void Linear3D<InputDataType, OutputDataType, RegularizerType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(inSize);
  ar & BOOST_SERIALIZATION_NVP(outSize);

  // This is inefficient, but we have to allocate this memory so that
  // WeightSetVisitor gets the right size.
  if (Archive::is_loading::value)
    weights.set_size(outSize * inSize + outSize, 1);
}

} // namespace ann
} // namespace mlpack

#endif
