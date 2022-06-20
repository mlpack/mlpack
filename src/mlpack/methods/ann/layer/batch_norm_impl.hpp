/**
 * @file methods/ann/layer/batch_norm_impl.hpp
 * @author Praveen Ch
 * @author Manthan-R-Sheth
 * @author Kartik Dutt
 * @author Shubham Agrawal
 *
 * Implementation of the Batch Normalization Layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_BatchNormType_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BatchNormType_IMPL_HPP

// In case it is not included.
#include "batch_norm.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename MatType>
BatchNormType<MatType>::BatchNormType() : 
    Layer<MatType>(true),
    size(0),
    eps(1e-8),
    average(true),
    momentum(0.0),
    count(0),
    averageFactor(0.0),
    inputDimension1(1),
    inputDimension2(1),
    higherDimension(1)
{
  // Nothing to do here.
}

template <typename MatType>
BatchNormType<MatType>::BatchNormType(
    const size_t size,
    const double eps,
    const bool average,
    const double momentum) : 
		Layer<MatType>(true),
    size(size),
    eps(eps),
    average(average),
    momentum(momentum),
    count(0),
    averageFactor(0.0),
    inputDimension1(1),
    inputDimension2(1),
    higherDimension(1)
{
  runningMean.zeros(size, 1);
  runningVariance.ones(size, 1);
}

// Copy constructor.
template<typename MatType>
BatchNormType<MatType>::BatchNormType(const BatchNormType& layer) :
    Layer<MatType>(layer),
    size(layer.size),
    eps(layer.eps),
    average(layer.average),
    momentum(layer.momentum),
    count(layer.count),
    averageFactor(layer.averageFactor),
    inputDimension1(layer.inputDimension1),
    inputDimension2(layer.inputDimension2),
    higherDimension(layer.higherDimension)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType>
BatchNormType<MatType>::BatchNormType(BatchNormType&& layer) :
    Layer<MatType>(std::move(layer)),
    size(std::move(layer.size)),
    eps(std::move(layer.eps)),
    average(std::move(layer.average)),
    momentum(std::move(layer.momentum)),
    count(std::move(layer.count)),
    averageFactor(std::move(layer.averageFactor)),
    inputDimension1(std::move(layer.inputDimension1)),
    inputDimension2(std.move(layer.inputDimension2)),
    higherDimension(std.move(layer.higherDimension))
{
  // Nothing else to do.
}

template<typename MatType>
BatchNormType<MatType>&
BatchNormType<MatType>::operator=(const BatchNormType& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(layer);
    size = layer.size;
    eps = layer.eps;
    average = layer.average;
    momentum = layer.momentum;
    count = layer.count;
    averageFactor = layer.averageFactor;
    inputDimension1 = layer.inputDimension1;
    inputDimension2 = layer.inputDimension2;
    higherDimension = layer.higherDimension;
  }

  return *this;
}

template<typename MatType>
BatchNormType<MatType>&
BatchNormType<MatType>::operator=(
    BatchNormType&& layer)
{
  if (&layer != this)
  {
    Layer<MatType>::operator=(std::move(layer));
    size = std::move(layer.size);
    eps = std::move(layer.eps);
    average = std::move(layer.average);
    momentum = std::move(layer.momentum);
    count = std::move(layer.count);
    averageFactor = std::move(layer.averageFactor);
    inputDimension1 = std::move(layer.inputDimension1);
    inputDimension2 = std.move(layer.inputDimension2);
    higherDimension = std.move(layer.higherDimension);
  }

  return *this;
}

template<typename MatType>
void BatchNormType<MatType>::SetWeights(
    typename MatType::elem_type* weightsPtr)
{
	MakeAlias(weights, weightsPtr, WeightSize(), 1);
  // Gamma acts as the scaling parameters for the normalized output.
	MakeAlias(gamma, weightsPtr, size, 1);
  // Beta acts as the shifting parameters for the normalized output.
	MakeAlias(beta, weightsPtr + gamma.n_elem, size, 1);
}

template<typename MatType>
void BatchNormType<MatType>::CustomInitialize(
    MatType& W,
    const size_t rows, 
    const size_t /* cols */)
{
  if (rows != 2*size) {
    throw std::invalid_argument("BatchNormType::CustomInitialize(): wrong "
        "rows size!"); 
  }
  MatType gamma_temp;
  MatType beta_temp;
  // Gamma acts as the scaling parameters for the normalized output.
	MakeAlias(gamma_temp, W.memptr(), size, 1);
  // Beta acts as the shifting parameters for the normalized output.
	MakeAlias(beta_temp, W.memptr() + gamma_temp.n_elem, size, 1);

  gamma_temp.fill(1.0);
  beta_temp.fill(0.0);
}

template<typename MatType>
void BatchNormType<MatType>::Forward(
    const MatType& input,
    MatType& output)
{
  const size_t batchSize = input.n_cols;
  const size_t inputSize = inputDimension1 * inputDimension2;

  // Set size of output equal to the size of input.
  // output.set_size(arma::size(input));

  // We will calculate minibatch norm on each channel / feature map.
  if (this->training)
  {
    // Check only during training, batch-size can be one during inference.
    if ((batchSize * higherDimension) == 1 && inputSize == 1)
    {
      Log::Warn << "Variance for single element isn't defined and" <<
          " will be set to  0.0 for training. Use a batch-size" <<
          " greater than 1 to fix the warning." << std::endl;
    }

    // Input corresponds to output from convolution layer.
    // Use a cube for simplicity.
    arma::Cube<typename MatType::elem_type> inputTemp(
        const_cast<MatType&>(input).memptr(), inputSize, size,
        batchSize * higherDimension, false, false);

    // Initialize output to same size and values for convenience.
    arma::Cube<typename MatType::elem_type> outputTemp(
        const_cast<MatType&>(output).memptr(), inputSize, size,
        batchSize * higherDimension, false, false);
    outputTemp = inputTemp;

    // Calculate mean and variance over all channels.
    mean = arma::mean(arma::mean(inputTemp, 2), 0);
    variance = arma::mean(arma::mean(arma::pow(
        inputTemp.each_slice() - arma::repmat(mean,
        inputSize, 1), 2), 2), 0);

    outputTemp.each_slice() -= arma::repmat(mean, inputSize, 1);

    // Used in backward propagation.
    inputMean.set_size(arma::size(inputTemp));
    inputMean = outputTemp;

    // Normalize output.
    outputTemp.each_slice() /= arma::sqrt(arma::repmat(variance,
        inputSize, 1) + eps);

    // Re-used in backward propagation.
    normalized.set_size(arma::size(inputTemp));
    normalized = outputTemp;

    outputTemp.each_slice() %= arma::repmat(gamma.t(),
        inputSize, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(),
        inputSize, 1);

    count += 1;
    averageFactor = average ? 1.0 / count : momentum;

    double nElements = 0.0;
    if (input.n_elem - size != 0)
      nElements = 1.0 / (input.n_elem - size + eps);

    // Update running mean and running variance.
    runningMean = (1 - averageFactor) * runningMean + averageFactor *
        mean.t();
    runningVariance = (1 - averageFactor) * runningVariance +
       input.n_elem * nElements *
       averageFactor * variance.t();
  }
  else
  {
    // Normalize the input and scale and shift the output.
    output = input;
    arma::Cube<typename MatType::elem_type> outputTemp(
        const_cast<MatType&>(output).memptr(), input.n_rows / size, size,
        batchSize * higherDimension, false, false);

    outputTemp.each_slice() -= arma::repmat(runningMean.t(),
        input.n_rows / size, 1);
    outputTemp.each_slice() /= arma::sqrt(arma::repmat(runningVariance.t(),
        input.n_rows / size, 1) + eps);
    outputTemp.each_slice() %= arma::repmat(gamma.t(),
        input.n_rows / size, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(),
        input.n_rows / size, 1);
  }
}

template<typename MatType>
void BatchNormType<MatType>::Backward(
    const MatType& input,
    const MatType& gy,
    MatType& g)
{
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  const size_t batchSize = input.n_cols;
  const size_t inputSize = inputDimension1 * inputDimension2;

  g.set_size(arma::size(input));
  arma::Cube<typename MatType::elem_type> gyTemp(
      const_cast<MatType&>(gy).memptr(), inputSize, size,
      batchSize * higherDimension, false, false);
  arma::Cube<typename MatType::elem_type> gTemp(
      const_cast<MatType&>(g).memptr(), inputSize, size,
      batchSize * higherDimension, false, false);

  // Step 1: dl / dxhat.
  arma::Cube<typename MatType::elem_type> norm =
      gyTemp.each_slice() % arma::repmat(gamma.t(), inputSize, 1);

  // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  MatType temp = arma::sum(norm % inputMean, 2);
  MatType vars = temp % arma::repmat(arma::pow(stdInv, 3),
      inputSize, 1) * -0.5;

  // Step 3: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  gTemp = (norm.each_slice() % arma::repmat(stdInv,
      inputSize, 1) +
      (inputMean.each_slice() % vars * 2)) / (batchSize * higherDimension);

  // Step 4: sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  MatType normTemp = arma::sum(norm.each_slice() %
      arma::repmat(-stdInv, inputSize, 1) , 2) /
      (batchSize * higherDimension);
  gTemp.each_slice() += normTemp;
}

template<typename MatType>
void BatchNormType<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  const size_t inputSize = inputDimension1 * inputDimension2;

  gradient.set_size(size + size, 1);
  arma::Cube<typename MatType::elem_type> errorTemp(
      const_cast<MatType&>(error).memptr(), inputSize, size,
      error.n_cols, false, false);

  // Step 5: dl / dy * xhat.
  MatType temp = arma::sum(arma::sum(normalized % errorTemp, 0), 2);
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = temp.t();

  // Step 6: dl / dy.
  temp = arma::sum(arma::sum(errorTemp, 0), 2);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = temp.t();
}

template<typename MatType>
void BatchNormType<MatType>::ComputeOutputDimensions()
{
  this->outputDimensions = this->inputDimensions;
  if (this->inputDimensions.size() == 1) 
  {
    if (size != this->inputDimensions[0]) 
    {
      Log::Fatal << "BatchNorm: input size (" << this->inputDimensions[0]
          << ") must be equal to the size (" << size << ") of the batch "
          << "normalization layer." << std::endl;
    }
    inputDimension1 = 1;
    inputDimension2 = 1;
    higherDimension = 1;
  } 
  else if (this->inputDimensions.size() == 2) 
  {
    if (size == 1) 
    {
      inputDimension1 = this->inputDimensions[0];
      inputDimension2 = this->inputDimensions[1];
      higherDimension = 1;
    } 
    else 
    {
      if (size != this->inputDimensions[1]) 
      {
        Log::Fatal << "BatchNorm: input size (" << this->inputDimensions[1]
            << ") must be equal to the size (" << size << ") of the batch "
            << "normalization layer." << std::endl;
      }
      inputDimension1 = this->inputDimensions[0];
      inputDimension2 = 1;
      higherDimension = 1;
    }
  } 
  else 
  {
    if (size != this->inputDimensions[2]) 
    {
      Log::Fatal << "BatchNorm: input size (" << this->inputDimensions[2]
          << ") must be equal to the size (" << size << ") of the batch "
          << "normalization layer." << std::endl;
    }
    inputDimension1 = this->inputDimensions[0];
    inputDimension2 = this->inputDimensions[1];
    higherDimension = 1;
    for (size_t i = 3; i < this->inputDimensions.size(); ++i)
    {
      higherDimension *= this->inputDimensions[i];
    }
  }
}

template<typename MatType>
template<typename Archive>
void BatchNormType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(count));
  ar(CEREAL_NVP(momentum));
  ar(CEREAL_NVP(average));
  ar(CEREAL_NVP(runningMean));
  ar(CEREAL_NVP(runningVariance));
  ar(CEREAL_NVP(inputMean));
  ar(CEREAL_NVP(inputDimension1));
  ar(CEREAL_NVP(inputDimension2));
  ar(CEREAL_NVP(higherDimension));
}

} // namespace ann
} // namespace mlpack

#endif
