/**
 * @file linear_recurrent_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of LinearRecurrent layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_RECURRENT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_RECURRENT_IMPL_HPP

#include "linear_recurrent.hpp"

namespace mlpack {

// Create the LinearRecurrent layer.
template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>::LinearRecurrentType() :
    RecurrentLayer<MatType>(),
    inSize(0),
    outSize(0)
{
  // Nothing to do.
}

template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>::LinearRecurrentType(
    const size_t outSize,
    RegularizerType regularizer) :
    RecurrentLayer<MatType>(),
    inSize(0), // This will be computed in ComputeOutputDimensions().
    outSize(outSize),
    regularizer(regularizer)
{
  // Nothing to do here.  Weights will be set later.
}

// Copy constructor.
template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>::LinearRecurrentType(
    const LinearRecurrentType& layer) :
    RecurrentLayer<MatType>(layer),
    inSize(layer.inSize),
    outSize(layer.outSize),
    regularizer(layer.regularizer)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>::LinearRecurrentType(
    LinearRecurrentType&& layer) :
    RecurrentLayer<MatType>(std::move(layer)),
    inSize(std::move(layer.inSize)),
    outSize(std::move(layer.outSize)),
    regularizer(std::move(layer.regularizer))
{
  // Reset parameters of other layer.
  layer.inSize = 0;
  layer.outSize = 0;
}

// Copy operator.
template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>&
LinearRecurrentType<MatType, RegularizerType>::operator=(
    const LinearRecurrentType& layer)
{
  if (&layer != this)
  {
    RecurrentLayer<MatType>::operator=(layer);
    inSize = layer.inSize;
    outSize = layer.outSize;
    regularizer = layer.regularizer;
  }

  return *this;
}

// Move operator.
template<typename MatType, typename RegularizerType>
LinearRecurrentType<MatType, RegularizerType>&
LinearRecurrentType<MatType, RegularizerType>::operator=(
    LinearRecurrentType&& layer)
{
  if (&layer != this)
  {
    RecurrentLayer<MatType>::operator=(std::move(layer));
    inSize = std::move(layer.inSize);
    outSize = std::move(layer.outSize);
    regularizer = std::move(layer.regularizer);

    // Reset parameters of other layer.
    layer.inSize = 0;
    layer.outSize = 0;
  }

  return *this;
}

// Set the parameters of the layer.
template<typename MatType, typename RegularizerType>
void LinearRecurrentType<MatType, RegularizerType>::SetWeights(
    const MatType& weightsIn)
{
  MakeAlias(parameters, weightsIn, WeightSize(), 1);
  MakeAlias(weights, weightsIn, outSize, inSize);
  MakeAlias(recurrentWeights, weightsIn, outSize, outSize, weights.n_elem);
  MakeAlias(bias, weightsIn, outSize, 1,
      weights.n_elem + recurrentWeights.n_elem);
}

// Forward pass of linear recurrent layer.
template<typename MatType, typename RegularizerType>
void LinearRecurrentType<MatType, RegularizerType>::Forward(
    const MatType& input, MatType& output)
{
  // Take the forward step: f(x) = Wx + Uh + b.
  if (!this->HasPreviousStep())
  {
    output = weights * input; // Omit the Uh term is there is no previous step.
  }
  else
  {
    output = weights * input +
        recurrentWeights * this->RecurrentState(this->PreviousStep());
  }

  #pragma omp for
  for (size_t c = 0; c < (size_t) output.n_cols; ++c)
    output.col(c) += bias;

  // Update the recurrent state if needed.
  if (!this->AtFinalStep())
    this->RecurrentState(this->CurrentStep()) = output;
}

// Backward pass of linear recurrent layer.
template<typename MatType, typename RegularizerType>
void LinearRecurrentType<MatType, RegularizerType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  // For the linear recurrent layer, we must compute the derivative of the
  // output with respect to the input through two different paths: via the
  // output, and via the hidden recurrence.

  // Via the output, the result is the same as for the regular linear layer.
  g = weights.t() * gy;

  // No need to go through the recurrence if this is the final time step.
  if (!this->AtFinalStep())
  {
    // Via the recurrence, the result is equivalent, just with the recurrent
    // gradient as the gy parameter.
    g += weights.t() * this->RecurrentGradient(this->CurrentStep());
  }

  if (this->HasPreviousStep())
  {
    // Compute the derivative of the recurrent state with respect to the output
    // (and the current hidden state, if this is not the last time step).
    //
    // With respect to the output, we can just propagate back through the
    // recurrent weights.
    this->RecurrentGradient(this->PreviousStep()) = recurrentWeights.t() * gy;

    if (!this->AtFinalStep())
    {
      // If we also have a path from dz/dh^t, this can be added.
      this->RecurrentGradient(this->PreviousStep()) +=
          recurrentWeights.t() * this->RecurrentGradient(this->CurrentStep());
    }
  }
}

// Compute the gradient with respect to the input.
template<typename MatType, typename RegularizerType>
void LinearRecurrentType<MatType, RegularizerType>::Gradient(
    const MatType& input,
    const MatType& error,
    MatType& gradient)
{
  // The derivative of the output layer with respect to the weights can be
  // computed as the sum through two paths: the current output, and the
  // recurrent connection.

  // First, through the output path.  This is the same as the non-recurrent
  // linear layer.
  //    dz/dW = dz/dy * x.t()
  //    dz/dU = dz/dy * d/dW (Wx + Uy_{t - 1} + b)
  //          = dz/dy * y_{t - 1}
  //    dz/db = dz/dy * 1
  //
  const size_t whOffset = weights.n_elem;
  const size_t bOffset = weights.n_elem + recurrentWeights.n_elem;
  gradient.submat(0, 0, whOffset - 1, 0) = vectorise(error * input.t());
  if (this->HasPreviousStep())
  {
    gradient.submat(whOffset, 0, bOffset - 1, 0) =
        vectorise(error * this->RecurrentState(this->PreviousStep()).t());
  }
  gradient.submat(bOffset, 0, gradient.n_rows - 1, 0) = sum(error, 1);

  // Now, through the hidden path.
  // The calculus is the same here, since the hidden path is the output, but
  // instead of using `error` (which is dz/dy) we use the hidden derivative
  // (which is dz/dh^{t - 1}).
  if (!this->AtFinalStep())
  {
    gradient.submat(0, 0, whOffset - 1, 0) +=
        vectorise(this->RecurrentGradient(this->CurrentStep()) * input.t());
    if (this->HasPreviousStep())
    {
      gradient.submat(whOffset, 0, bOffset - 1, 0) +=
          vectorise(this->RecurrentGradient(this->CurrentStep()) *
                    this->RecurrentState(this->PreviousStep()).t());
    }
    gradient.submat(bOffset, 0, gradient.n_rows - 1, 0) += sum(
        this->RecurrentGradient(this->CurrentStep()), 1);

    // this->HiddenDeriv(this->PreviousStep()) was already computed in
    // Backward(), so no need to do it here.
  }
}

// Get the total number of trainable parameters.
template<typename MatType, typename RegularizerType>
size_t LinearRecurrentType<MatType, RegularizerType>::WeightSize() const
{
  return (inSize * outSize) /* weight matrix */ +
      (outSize * outSize) /* recurrent state matrix */ +
      outSize /* bias vector */;
}

template<typename MatType, typename RegularizerType>
size_t LinearRecurrentType<MatType, RegularizerType>::RecurrentSize() const
{
  return outSize;
}

// Compute the output dimensions of the layer, assuming that inputDimension has
// been set.
template<typename MatType, typename RegularizerType>
void LinearRecurrentType<MatType, RegularizerType>::ComputeOutputDimensions()
{
  // Compute the total number of input dimensions.
  inSize = this->inputDimensions[0];
  for (size_t i = 1; i < this->inputDimensions.size(); ++i)
    inSize *= this->inputDimensions[i];

  this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(), 1);

  // The LinearRecurrent layer flattens its input.
  this->outputDimensions[0] = outSize;
}

// Serialize the layer.
template<typename MatType, typename RegularizerType>
template<typename Archive>
void LinearRecurrentType<MatType, RegularizerType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<RecurrentLayer<MatType>>(this));

  ar(CEREAL_NVP(inSize));
  ar(CEREAL_NVP(outSize));
  ar(CEREAL_NVP(regularizer));
}

} // namespace mlpack

#endif
