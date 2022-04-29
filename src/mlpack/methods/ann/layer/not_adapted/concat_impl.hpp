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
namespace ann /** Artificial Neural Network. */ {

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::ConcatType(
    const bool run) :
    axis(0),
    useAxis(false)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::ConcatType(
    const size_t axis,
    const bool run) :
    axis(axis),
    useAxis(true)
{
  // Nothing to do.
}

template<typename InputType, typename OutputType>
ConcatType<InputType, OutputType>::~ConcatType()
{
  // Clear memory.
  for (size_t i = 0; i < this->network.size(); ++i)
    delete this->network[i];
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Forward(
    const InputType& input, OutputType& output)
{
  this->InitializeForwardPassMemory();

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

  size_t slices = (axis == 0) ? input.n_cols :
      std::accumulate(this->outputDimensions.begin(),
          this->outputDimensions.begin() + axis, 0) + input.n_cols;
  size_t rows = (axis == this->outputDimensions.size() - 1) ? 1 :
      std::accumulate(this->outputDimensions.begin() + axis + 1,
          this->outputDimensions.end(), 0);

  std::vector<arma::Cube<typename OutputType::elem_type>> layerOutputAliases;
  for (size_t i = 0; i < this->layerOutputs.size(); ++i)
  {
    layerOutputAliases.emplace_back(arma::Cube<typename OutputType::elem_type>(
        this->layerOutputs[i].memptr(), rows,
        this->network[i]->OutputDimensions()[axis], slices, false, true);
  }

  arma::Cube<typename OutputType::elem_type> output(output.memptr(), rows,
      this->outputDimensions[axis], slices, false, true);

  // Now get the columns from each output.
  size_t startCol = 0;
  for (size_t i = 0; i < layerOutputAliases.size(); ++i)
  {
    const size_t cols = layerOutputAliases[i].n_cols;
    output.cols(startCol, startCol + cols - 1) = layerOutputAliases[i];
    startCol += cols;
  }
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Backward(
    const InputType& /* input */, const OutputType& gy, OutputType& g)
{
  this->InitializeBackwardPassMemory();

  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `gy` to the layers.

  size_t slices = (axis == 0) ? gy.n_cols :
      std::accumulate(this->outputDimensions.begin(),
          this->outputDimensions.begin() + axis, 0) + gy.n_cols;
  size_t rows = (axis == this->outputDimensions.size() - 1) ? 1 :
      std::accumulate(this->outputDimensions.begin() + axis + 1,
          this->outputDimensions.end(), 0);

  arma::Cube<typename OutputType::elem_type> gyTmp(gy.memptr(), rows,
      this->outputDimensions[axis], slices, false, true);

  size_t startCol = 0;
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    const size_t cols = this->network[i]->OutputDimensions()[axis];
    // TODO: is delta size correct?
    // TODO: no copy!
    OutputType delta = gyTmp.cols(startCol, startCol + cols - 1);
    // TODO: consider batch size correctly
    delta.reshape( ... );
    this->network[i]->Backward(this->layerOutputs[i], delta,
        this->layerDeltas[i]);

    startCol += cols;
  }

  g = this->layerDeltas[0];
  for (size_t i = 1; i < this->network.size(); ++i)
  {
    g += this->layerDeltas[i];
  }
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Backward(
    const InputType& /* input */,
    const OutputType& gy,
    OutputType& g,
    const size_t index)
{
  // We only intend to perform a backward pass on one layer.
  // Thus, we need to extract the parts of gy that correspond to the desired
  // layer (specified by `index`).

  size_t slices = (axis == 0) ? gy.n_cols :
      std::accumulate(this->outputDimensions.begin(),
          this->outputDimensions.begin() + axis, 0) + gy.n_cols;
  size_t rows = (axis == this->outputDimensions.size() - 1) ? 1 :
      std::accumulate(this->outputDimensions.begin() + axis + 1,
          this->outputDimensions.end(), 0);

  arma::Cube<typename OutputType::elem_type> gyTmp(gy.memptr(), rows,
      this->outputDimensions[axis], slices, false, true);

  size_t startCol = 0;
  for (size_t i = 0; i < index; ++i)
  {
    startCol += this->network[i]->OutputDimensions()[axis];
  }

  // TODO: no copy!
  const size_t cols = this->network[index]->OutputDimensions()[axis];
  OutputType delta = gyTmp.cols(startCol, startCol + cols - 1);
  delta.reshape( ... );

  this->network[index]->Backward(this->layerOutputs[index], delta, g);
}

template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient)
{
  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `gy` to the layers.

  size_t slices = (axis == 0) ? input.n_cols :
      std::accumulate(this->outputDimensions.begin(),
          this->outputDimensions.begin() + axis, 0) + input.n_cols;
  size_t rows = (axis == this->outputDimensions.size() - 1) ? 1 :
      std::accumulate(this->outputDimensions.begin() + axis + 1,
          this->outputDimensions.end(), 0);

  arma::Cube<typename OutputType::elem_type> errorTmp(error.memptr(), rows,
      this->outputDimensions[axis], slices, false, true);

  size_t startCol = 0;
  size_t startParam = 0;
  for (size_t i = 0; i < this->network.size(); ++i)
  {
    const size_t cols = this->network[i]->OutputDimensions()[axis];
    const size_t params = this->network[i]->WeightSize();

    OutputType err = errorTmp.cols(startCol, startCol + cols - 1);
    err.reshape(input.n_cols, err.n_elem / input.n_cols);
    // TODO: what about layerGradients?
    OutputType gradientAlias(gradient.colptr(startParam, 1, params, false,
        true);
    this->network[i]->Gradient(input, err, gradientAlias);

    startCol += cols;
    startParam += params;
  }
}

// TODO: adapt
template<typename InputType, typename OutputType>
void ConcatType<InputType, OutputType>::Gradient(
    const InputType& input,
    const OutputType& error,
    OutputType& gradient,
    const size_t index)
{
  // Just like the forward pass, we can treat our inputs as a cube, but here we
  // have to distribute the correct parts of `gy` to the layers.

  size_t slices = (axis == 0) ? input.n_cols :
      std::accumulate(this->outputDimensions.begin(),
          this->outputDimensions.begin() + axis, 0) + input.n_cols;
  size_t rows = (axis == this->outputDimensions.size() - 1) ? 1 :
      std::accumulate(this->outputDimensions.begin() + axis + 1,
          this->outputDimensions.end(), 0);

  arma::Cube<typename OutputType::elem_type> errorTmp(error.memptr(), rows,
      this->outputDimensions[axis], slices, false, true);

  size_t startCol = 0;
  size_t startParam = 0;
  for (size_t i = 0; i < index; ++i)
  {
    startCol += this->network[i]->OutputDimensions()[axis];
    startParam += this->network[i]->WeightSize();
  }

  const size_t cols = this->network[index]->OutputDimensions()[axis];
  const size_t params = this->network[index]->WeightSize();

  // TODO: no copy!
  OutputType err = errorTmp.cols(startCol, startCol + cols - 1);
  err.reshape(input.n_cols, err.n_elem / input.n_cols);
  OutputType gradientAlias(gradient.memptr(), 1, params, false, true);
  this->network[index]->Gradient(input, err, gradientAlias);
}

template<typename InputType, typename OutputType>
template<typename Archive>
void ConcatType<InputType, OutputType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<InputType, OutputType>>(this));

  ar(CEREAL_NVP(axis));
  ar(CEREAL_NVP(useAxis));
}

} // namespace ann
} // namespace mlpack

#endif
