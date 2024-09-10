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

#ifndef MLPACK_METHODS_ANN_LAYER_BATCHNORM_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_BATCHNORM_IMPL_HPP

// In case it is not included.
#include "batch_norm.hpp"

namespace mlpack {

template<typename MatType>
BatchNormType<MatType>::BatchNormType() :
    Layer<MatType>(),
    minAxis(2),
    maxAxis(2),
    eps(1e-8),
    average(true),
    momentum(0.0),
    count(0),
    inputDimension(1),
    size(0),
    higherDimension(1)
{
  // Nothing to do here.
}

template <typename MatType>
BatchNormType<MatType>::BatchNormType(
    const size_t minAxis,
    const size_t maxAxis,
    const double eps,
    const bool average,
    const double momentum) :
    Layer<MatType>(),
    minAxis(minAxis),
    maxAxis(maxAxis),
    eps(eps),
    average(average),
    momentum(momentum),
    count(0),
    inputDimension(1),
    size(0),
    higherDimension(1)
{
  // Nothing to do here.
}

// Copy constructor.
template<typename MatType>
BatchNormType<MatType>::BatchNormType(const BatchNormType& layer) :
    Layer<MatType>(layer),
    minAxis(layer.minAxis),
    maxAxis(layer.maxAxis),
    eps(layer.eps),
    average(layer.average),
    momentum(layer.momentum),
    variance(layer.variance),
    count(layer.count),
    inputDimension(layer.inputDimension),
    size(layer.size),
    higherDimension(layer.higherDimension),
    runningMean(layer.runningMean),
    runningVariance(layer.runningVariance)
{
  // Nothing else to do.
}

// Move constructor.
template<typename MatType>
BatchNormType<MatType>::BatchNormType(BatchNormType&& layer) :
    Layer<MatType>(std::move(layer)),
    minAxis(std::move(layer.minAxis)),
    maxAxis(std::move(layer.maxAxis)),
    eps(std::move(layer.eps)),
    average(std::move(layer.average)),
    momentum(std::move(layer.momentum)),
    variance(std::move(layer.variance)),
    count(std::move(layer.count)),
    inputDimension(std::move(layer.inputDimension)),
    size(std::move(layer.size)),
    higherDimension(std::move(layer.higherDimension)),
    runningMean(std::move(layer.runningMean)),
    runningVariance(std::move(layer.runningVariance))
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
    minAxis = layer.minAxis;
    maxAxis = layer.maxAxis;
    eps = layer.eps;
    average = layer.average;
    momentum = layer.momentum;
    variance = layer.variance;
    count = layer.count;
    inputDimension = layer.inputDimension;
    size = layer.size;
    higherDimension = layer.higherDimension;
    runningMean = layer.runningMean;
    runningVariance = layer.runningVariance;
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
    minAxis = std::move(layer.minAxis);
    maxAxis = std::move(layer.maxAxis);
    eps = std::move(layer.eps);
    average = std::move(layer.average);
    momentum = std::move(layer.momentum);
    variance = std::move(layer.variance);
    count = std::move(layer.count);
    inputDimension = std::move(layer.inputDimension);
    size = std::move(layer.size);
    higherDimension = std::move(layer.higherDimension);
    runningMean = std::move(layer.runningMean);
    runningVariance = std::move(layer.runningVariance);
  }

  return *this;
}

template<typename MatType>
void BatchNormType<MatType>::SetWeights(const MatType& weightsIn)
{
  MakeAlias(weights, weightsIn, WeightSize(), 1);
  // Gamma acts as the scaling parameters for the normalized output.
  MakeAlias(gamma, weightsIn, size, 1);
  // Beta acts as the shifting parameters for the normalized output.
  MakeAlias(beta, weightsIn, size, 1, gamma.n_elem);
}

template<typename MatType>
void BatchNormType<MatType>::CustomInitialize(
    MatType& W,
    const size_t elements)
{
  if (elements != 2 * size) {
    throw std::invalid_argument("BatchNormType::CustomInitialize(): wrong "
        "elements size!");
  }
  MatType gammaTemp;
  MatType betaTemp;
  // Gamma acts as the scaling parameters for the normalized output.
  MakeAlias(gammaTemp, W, size, 1);
  // Beta acts as the shifting parameters for the normalized output.
  MakeAlias(betaTemp, W, size, 1, gammaTemp.n_elem);

  gammaTemp.fill(1.0);
  betaTemp.fill(0.0);

  runningMean.zeros(size, 1);
  runningVariance.ones(size, 1);
}

template<typename MatType>
void BatchNormType<MatType>::Forward(
    const MatType& input,
    MatType& output)
{
  const size_t batchSize = input.n_cols;
  const size_t inputSize = inputDimension;
  const size_t m = inputSize * batchSize * higherDimension;

  // We will calculate minibatch norm on each channel / feature map.
  if (this->training)
  {
    // Check only during training, batch-size can be one during inference.
    if ((batchSize * higherDimension) == 1 && inputSize == 1)
    {
      Log::Warn << "Variance for single element isn't defined and" <<
          " will be set to 0.0 for training. Use a batch-size" <<
          " greater than 1 to fix the warning." << std::endl;
    }

    // Input corresponds to output from previous layer.
    // Used a cube for simplicity.
    arma::Cube<typename MatType::elem_type> inputTemp(
        const_cast<MatType&>(input).memptr(), inputSize, size,
        batchSize * higherDimension, false, false);

    // Initialize output to same size and values for convenience.
    arma::Cube<typename MatType::elem_type> outputTemp(
        const_cast<MatType&>(output).memptr(), inputSize, size,
        batchSize * higherDimension, false, false);
    outputTemp = inputTemp;

    // Calculate mean and variance over all channels.
    MatType mean = sum(sum(inputTemp, 2), 0) / m;
    variance = sum(sum(pow(
        inputTemp.each_slice() - repmat(mean, inputSize, 1), 2), 2), 0) / m;

    outputTemp.each_slice() -= repmat(mean, inputSize, 1);

    // Used in backward propagation.
    inputMean.set_size(arma::size(inputTemp));
    inputMean = outputTemp;

    // Normalize output.
    outputTemp.each_slice() /= sqrt(repmat(variance, inputSize, 1) + eps);

    // Re-used in backward propagation.
    normalized.set_size(arma::size(inputTemp));
    normalized = outputTemp;

    outputTemp.each_slice() %= repmat(gamma.t(), inputSize, 1);
    outputTemp.each_slice() += repmat(beta.t(), inputSize, 1);

    count += 1;
    // Value for average factor which used to update running parameters.
    double averageFactor = average ? 1.0 / count : momentum;

    double nElements = 0.0;
    if (m - 1 != 0)
      nElements = m * (1.0 / (m - 1));

    // Update running mean and running variance.
    runningMean = (1 - averageFactor) * runningMean + averageFactor *
        mean.t();
    runningVariance = (1 - averageFactor) * runningVariance +
        nElements * averageFactor * variance.t();
  }
  else
  {
    // Normalize the input and scale and shift the output.
    output = input;
    arma::Cube<typename MatType::elem_type> outputTemp(
        const_cast<MatType&>(output).memptr(), inputSize, size,
        batchSize * higherDimension, false, false);

    outputTemp.each_slice() -= repmat(runningMean.t(), inputSize, 1);
    outputTemp.each_slice() /= sqrt(repmat(runningVariance.t(),
        inputSize, 1) + eps);
    outputTemp.each_slice() %= repmat(gamma.t(), inputSize, 1);
    outputTemp.each_slice() += repmat(beta.t(), inputSize, 1);
  }
}

template<typename MatType>
void BatchNormType<MatType>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  const MatType stdInv = 1.0 / sqrt(variance + eps);

  const size_t batchSize = gy.n_cols;
  const size_t inputSize = inputDimension;
  const size_t m = inputSize * batchSize * higherDimension;

  arma::Cube<typename MatType::elem_type> gyTemp(
      const_cast<MatType&>(gy).memptr(), inputSize, size,
      batchSize * higherDimension, false, false);
  arma::Cube<typename MatType::elem_type> gTemp(
      const_cast<MatType&>(g).memptr(), inputSize, size,
      batchSize * higherDimension, false, false);

  // Step 1: dl / dxhat.
  arma::Cube<typename MatType::elem_type> norm =
      gyTemp.each_slice() % repmat(gamma.t(), inputSize, 1);

  // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  MatType temp = sum(sum(norm % inputMean, 2), 0);
  MatType vars = temp % pow(stdInv, 3) * (-0.5);

  // Step 3: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  gTemp = (norm.each_slice() % repmat(stdInv, inputSize, 1)) +
      ((inputMean.each_slice() % repmat(vars, inputSize, 1) * 2.0) / m);

  // Step 4: sum (dl / dxhat * -1 / stdInv) + variance *
  // sum (-2 * (x - mu)) / m.
  MatType normTemp = sum(sum((norm.each_slice() %
      repmat(-stdInv, inputSize, 1)) +
      (inputMean.each_slice() % repmat(vars, inputSize, 1) * (-2.0) / m),
      2), 0) / m;
  gTemp.each_slice() += repmat(normTemp, inputSize, 1);
}

template<typename MatType>
void BatchNormType<MatType>::Gradient(
    const MatType& /* input */,
    const MatType& error,
    MatType& gradient)
{
  const size_t inputSize = inputDimension;

  arma::Cube<typename MatType::elem_type> errorTemp(
      const_cast<MatType&>(error).memptr(), inputSize, size,
      error.n_cols * higherDimension, false, false);

  // Step 5: dl / dy * xhat.
  MatType temp = sum(sum(normalized % errorTemp, 0), 2);
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = temp.t();

  // Step 6: dl / dy.
  temp = sum(sum(errorTemp, 0), 2);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = temp.t();
}

template<typename MatType>
void BatchNormType<MatType>::ComputeOutputDimensions()
{
  if (minAxis > maxAxis)
  {
    Log::Fatal << "BatchNorm: minAxis must be less than or equal to maxAxis."
        << std::endl;
  }
  this->outputDimensions = this->inputDimensions;
  size_t mainMinAxis = minAxis;
  if (minAxis > this->inputDimensions.size() - 1)
  {
    Log::Warn << "BatchNorm: minAxis is out of range.  Setting to last axis."
        << std::endl;
    mainMinAxis = this->inputDimensions.size() - 1;
  }

  size_t mainMaxAxis = maxAxis;
  if (maxAxis > this->inputDimensions.size() - 1)
  {
    Log::Warn << "BatchNorm: maxAxis is out of range.  Setting to last axis."
        << std::endl;
    mainMaxAxis = this->inputDimensions.size() - 1;
  }

  inputDimension = 1;
  for (size_t i = 0; i < mainMinAxis; i++)
    inputDimension *= this->inputDimensions[i];

  size = this->inputDimensions[mainMinAxis];
  for (size_t i = mainMinAxis + 1; i <= mainMaxAxis; i++)
    size *= this->inputDimensions[i];

  higherDimension = 1;
  for (size_t i = mainMaxAxis + 1; i < this->inputDimensions.size(); i++)
    higherDimension *= this->inputDimensions[i];
}

template<typename MatType>
template<typename Archive>
void BatchNormType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<Layer<MatType>>(this));

  ar(CEREAL_NVP(minAxis));
  ar(CEREAL_NVP(maxAxis));
  ar(CEREAL_NVP(eps));
  ar(CEREAL_NVP(count));
  ar(CEREAL_NVP(momentum));
  ar(CEREAL_NVP(average));
  ar(CEREAL_NVP(runningMean));
  ar(CEREAL_NVP(runningVariance));
  ar(CEREAL_NVP(inputMean));
  ar(CEREAL_NVP(inputDimension));
  ar(CEREAL_NVP(size));
  ar(CEREAL_NVP(higherDimension));
}

} // namespace mlpack

#endif
