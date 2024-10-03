/**
 * @file methods/ann/layer/concat_impl.hpp
 * @author Marcus Edel
 * @author Mehul Kumar Nirala
 *
 * Implementation of the Concat class, which acts as a concatenation contain.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_IMPL_HPP

// In case it hasn't yet been included.
#include "concat.hpp"

namespace mlpack {

template<typename MatType>
ConcatType<MatType>::ConcatType(
    const size_t axis) :
    MultiLayer<MatType>(),
    axis(axis),
    useAxis(true)
{
  // Nothing to do.
}

template<typename MatType>
ConcatType<MatType>::ConcatType() :
    MultiLayer<MatType>(),
    axis(0),
    useAxis(false)
{
  // Nothing to do.
}

template<typename MatType>
ConcatType<MatType>::~ConcatType()
{
  // Nothing to do: the child layer memory is already cleared by MultiLayer.
}

template<typename MatType>
ConcatType<MatType>::ConcatType(const ConcatType& other) :
    MultiLayer<MatType>(other),
    axis(other.axis),
    useAxis(other.useAxis)
{
  // Nothing else to do.
}

template<typename MatType>
ConcatType<MatType>::ConcatType(ConcatType&& other) :
    MultiLayer<MatType>(std::move(other)),
    axis(std::move(other.axis)),
    useAxis(std::move(other.useAxis))
{
  // Nothing else to do.
}

template<typename MatType>
ConcatType<MatType>& ConcatType<MatType>::operator=(const ConcatType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
    axis = other.axis;
    useAxis = other.useAxis;
  }

  return *this;
}

template<typename MatType>
ConcatType<MatType>& ConcatType<MatType>::operator=(ConcatType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(std::move(other));
    axis = std::move(other.axis);
    useAxis = std::move(other.useAxis);
  }

  return *this;
}

template<typename MatType>
void ConcatType<MatType>::Forward(const MatType& input, MatType& output)
{
  // The implementation of MultiLayer is fine: this will allocate a matrix that
  // is able to hold each child layer's output.
  this->InitializeForwardPassMemory(input.n_cols);

  // Pass the input through all the layers in the network.
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    this->network[i]->Forward(input, this->layerOutputs[i]);
  }

  // Now concatenate the outputs along the correct axis.
  // We can actually use Armadillo to do this for us---we will treat the axis of
  // interest as "columns", any axes that come before the axis of interest as
  // 'flattened slices', and any axes that come after the axis of interest as
  // 'flattened rows'.  As a result, we will only have to do join_cols() to
  // produce the right result.
  //
  // Note that we will have one "extra" axis in addition to
  // this->outputDimensions.size(); that is the batch size (represented as the
  // number of columns in `input`).

  size_t rows = 1;
  for (size_t i = 0; i < axis; ++i)
    rows *= this->outputDimensions[i];

  size_t slices = input.n_cols;
  for (size_t i = axis + 1; i < this->outputDimensions.size(); ++i)
    slices *= this->outputDimensions[i];

  std::vector<arma::Cube<typename MatType::elem_type>> layerOutputAliases(
      this->layerOutputs.size());
  for (size_t i = 0; i < this->layerOutputs.size(); ++i)
  {
    MakeAlias(layerOutputAliases[i], this->layerOutputs[i], rows,
        this->network[i]->OutputDimensions()[axis], slices);
  }

  arma::Cube<typename MatType::elem_type> outputAlias;
  MakeAlias(outputAlias, output, rows, this->outputDimensions[axis], slices);

  // Now get the columns from each output.
  size_t startCol = 0;
  for (size_t i = 0; i < layerOutputAliases.size(); ++i)
  {
    const size_t cols = layerOutputAliases[i].n_cols;
    outputAlias.cols(startCol, startCol + cols - 1) = layerOutputAliases[i];
    startCol += cols;
  }
}

template<typename MatType>
void ConcatType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // The implementation of MultiLayer is fine: this will allocate a matrix that
  // is able to hold each child layer's delta (which has the same size as the
  // input).
  this->InitializeBackwardPassMemory(gy.n_cols);

  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `gy` to the layers.

  size_t rows = 1;
  for (size_t i = 0; i < axis; ++i)
    rows *= this->outputDimensions[i];

  size_t slices = gy.n_cols;
  for (size_t i = axis + 1; i < this->outputDimensions.size(); ++i)
    slices *= this->outputDimensions[i];

  arma::Cube<typename MatType::elem_type> gyTmp;
  MakeAlias(gyTmp, gy, rows, this->outputDimensions[axis], slices);

  size_t startCol = 0;
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    const size_t cols = this->network[i]->OutputDimensions()[axis];
    MatType delta = gyTmp.cols(startCol, startCol + cols - 1);
    // Reshape so that the batch size is the number of columns.
    delta.reshape(delta.n_elem / gy.n_cols, gy.n_cols);
    this->network[i]->Backward(
        input,
        this->layerOutputs[i],
        delta,
        this->layerDeltas[i]);

    startCol += cols;
  }

  g = this->layerDeltas[0];
  for (size_t i = 1; i < this->network.size(); ++i)
  {
    g += this->layerDeltas[i];
  }
}

template<typename MatType>
void ConcatType<MatType>::Backward(
    const MatType& input,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g,
    const size_t index)
{
  // We only intend to perform a backward pass on one layer.
  // Thus, we need to extract the parts of gy that correspond to the desired
  // layer (specified by `index`).

  size_t rows = 1;
  for (size_t i = 0; i < axis; ++i)
    rows *= this->outputDimensions[i];

  size_t slices = gy.n_cols;
  for (size_t i = axis + 1; i < this->outputDimensions.size(); ++i)
    slices *= this->outputDimensions[i];

  arma::Cube<typename MatType::elem_type> gyTmp;
  MakeAlias(gyTmp, gy, rows, this->outputDimensions[axis], slices);

  size_t startCol = 0;
  for (size_t i = 0; i < index; ++i)
  {
    startCol += this->network[i]->OutputDimensions()[axis];
  }

  const size_t cols = this->network[index]->OutputDimensions()[axis];
  MatType delta = gyTmp.cols(startCol, startCol + cols - 1);
  // Reshape so that the batch size is the number of columns.
  delta.reshape(delta.n_elem / gy.n_cols, gy.n_cols);

  this->network[index]->Backward(input, this->layerOutputs[index], delta, g);
}

template<typename MatType>
void ConcatType<MatType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `error` to the layers.

  size_t rows = 1;
  for (size_t i = 0; i < axis; ++i)
    rows *= this->outputDimensions[i];

  size_t slices = input.n_cols;
  for (size_t i = axis + 1; i < this->outputDimensions.size(); ++i)
    slices *= this->outputDimensions[i];

  arma::Cube<typename MatType::elem_type> errorTmp;
  MakeAlias(errorTmp, error, rows, this->outputDimensions[axis], slices);

  size_t startCol = 0;
  size_t startParam = 0;
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    const size_t cols = this->network[i]->OutputDimensions()[axis];
    const size_t params = this->network[i]->WeightSize();

    MatType err = errorTmp.cols(startCol, startCol + cols - 1);
    err.reshape(err.n_elem / input.n_cols, input.n_cols);
    MatType gradientAlias;
    MakeAlias(gradientAlias, gradient, params, 1, startParam);
    this->network[i]->Gradient(input, err, gradientAlias);

    startCol += cols;
    startParam += params;
  }
}

template<typename MatType>
void ConcatType<MatType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient,
    const size_t index)
{
  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `error` to the layers.

  size_t rows = 1;
  for (size_t i = 0; i < axis; ++i)
    rows *= this->outputDimensions[i];

  size_t slices = input.n_cols;
  for (size_t i = axis + 1; i < this->outputDimensions.size(); ++i)
    slices *= this->outputDimensions[i];

  arma::Cube<typename MatType::elem_type> errorTmp;
  MakeAlias(errorTmp, error, rows, this->outputDimensions[axis], slices);

  size_t startCol = 0;
  size_t startParam = 0;
  for (size_t i = 0; i < index; ++i)
  {
    startCol += this->network[i]->OutputDimensions()[axis];
    startParam += this->network[i]->WeightSize();
  }

  const size_t cols = this->network[index]->OutputDimensions()[axis];
  const size_t params = this->network[index]->WeightSize();

  MatType err = errorTmp.cols(startCol, startCol + cols - 1);
  err.reshape(err.n_elem / input.n_cols, input.n_cols);
  MatType gradientAlias;
  MakeAlias(gradientAlias, gradient, params, 1, startParam);
  this->network[index]->Gradient(input, err, gradientAlias);
}

template<typename MatType>
template<typename Archive>
void ConcatType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(useAxis));
}

} // namespace mlpack

#endif
