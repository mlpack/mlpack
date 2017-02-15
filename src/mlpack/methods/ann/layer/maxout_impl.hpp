/**
 * @file maxout_impl.hpp
 * @author Prasanna Patil
 *
 * Implementation of the Maxout layer class also known as fully-connected layer
 * or affine transformation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAXOUT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_MAXOUT_IMPL_HPP

// In case it hasn't yet been included.
#include "maxout.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
Maxout<InputDataType, OutputDataType>::Maxout()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
Maxout<InputDataType, OutputDataType>::Maxout(
    const size_t inSize,
    const size_t hiddenSize,
    const size_t outSize) :
    inSize(inSize),
    hiddenSize(hiddenSize),
    outSize(outSize)
{
  weights.set_size((inSize * hiddenSize * outSize) + (outSize * hiddenSize), 1);
}

template<typename InputDataType, typename OutputDataType>
void Maxout<InputDataType, OutputDataType>::Reset()
{
  weight = arma::mat(weights.memptr(), outSize * hiddenSize, inSize,
      false, false);
  bias = arma::mat(weights.memptr() + weight.n_elem,
      outSize * hiddenSize, 1, false, false);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Maxout<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
{
  if (hiddenParameter.n_elem == 0) {
    hiddenParameter = arma::zeros<arma::mat>(outSize * hiddenSize,
        input.n_cols);
  }

  if (output.n_elem == 0) {
    output = arma::zeros<arma::mat>(outSize, input.n_cols);
  }

  hiddenParameter = (weight * input) + bias;

  for (int i = 0, j = 0; j < outSize; i+=hiddenSize, j++) {
    output.row(j) = arma::max(hiddenParameter.rows(i, i + hiddenSize - 1), 0);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Maxout<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  arma::mat hiddenGradient = arma::zeros<arma::mat>(hiddenParameter.n_rows,
      hiddenParameter.n_cols);
  arma::mat hiddenParameterSubset;

  for (int i = 0, j = 0; j < outSize; i+=hiddenSize, j++) {
    hiddenParameterSubset = hiddenParameter.rows(i, i + hiddenSize - 1);
    
    for (int k = 0; k < hiddenParameterSubset.n_cols; k++) {
      int ind = arma::as_scalar(arma::find(arma::max(
          hiddenParameterSubset.col(k)) == hiddenParameterSubset.col(k)));
      hiddenGradient(i + ind, k) = gy(j, k);
    }
  }

  g = weight.t() * hiddenGradient;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void Maxout<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>&& input,
    arma::Mat<eT>&& error,
    arma::Mat<eT>&& gradient)
{
  arma::mat hiddenGradient = arma::zeros<arma::mat>(hiddenParameter.n_rows,
      hiddenParameter.n_cols);
  arma::mat hiddenParameterSubset;

  for (int i = 0, j = 0; j < outSize; i+=hiddenSize, j++) {
    hiddenParameterSubset = hiddenParameter.rows(i, i + hiddenSize - 1);
    
    for (int k = 0; k < hiddenParameterSubset.n_cols; k++) {
      int ind = arma::as_scalar(arma::find(arma::max(
          hiddenParameterSubset.col(k)) == hiddenParameterSubset.col(k)));
      hiddenGradient(i + ind, k) = error(j, k);
    }
  }

  gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
      hiddenGradient * input.t() / input.n_cols);

  arma::colvec bias_ones = arma::ones<arma::colvec>(input.n_cols);
  gradient.submat(weight.n_elem, 0, gradient.n_elem - 1, 0) = (
    hiddenGradient * bias_ones / input.n_cols);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Maxout<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(weights, "weights");
  ar & data::CreateNVP(inSize, "inSize");
  ar & data::CreateNVP(hiddenSize, "hiddenSize");
  ar & data::CreateNVP(outSize, "outSize");
}

} // namespace ann
} // namespace mlpack

#endif
