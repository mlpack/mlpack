/**
 * @file methods/ann/layer/batch_norm_impl.hpp
 * @author Praveen Ch
 * @author Manthan-R-Sheth
 * @author Kartik Dutt
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
namespace ann { /** Artificial Neural Network. */

template<typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm() :
    size(0),
    eps(1e-8),
    average(true),
    momentum(0.0),
    loading(false),
    deterministic(false),
    count(0),
    averageFactor(0.0)
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
BatchNorm<InputDataType, OutputDataType>::BatchNorm(
    const size_t size,
    const double eps,
    const bool average,
    const double momentum) :
    size(size),
    eps(eps),
    average(average),
    momentum(momentum),
    loading(false),
    deterministic(false),
    count(0),
    averageFactor(0.0)
{
  weights.set_size(size + size, 1);
  runningMean.zeros(size, 1);
  runningVariance.ones(size, 1);
}

template<typename InputDataType, typename OutputDataType>
void BatchNorm<InputDataType, OutputDataType>::Reset()
{
  // Gamma acts as the scaling parameters for the normalized output.
  gamma = arma::mat(weights.memptr(), size, 1, false, false);
  // Beta acts as the shifting parameters for the normalized output.
  beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);

  if (!loading)
  {
    gamma.fill(1.0);
    beta.fill(0.0);
  }

  deterministic = false;
  loading = false;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  Log::Assert(input.n_rows % size == 0, "Input features must be divisible \
      by feature maps.");

  const size_t batchSize = input.n_cols;
  const size_t inputSize = input.n_rows / size;

  // Set size of output equal to the size of input.
  output.set_size(arma::size(input));

  // We will calculate minibatch norm on each channel / feature map.
  if (!deterministic)
  {
    // Check only during training, batch-size can be one during inference.
    if (batchSize == 1 && inputSize == 1)
    {
      Log::Warn << "Variance for single element isn't defined and" <<
          " will be set to  0.0 for training. Use a batch-size" <<
          " greater than 1 to fix the warning." << std::endl;
    }

    // Input corresponds to output from convolution layer.
    // Use a cube for simplicity.
    arma::cube inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(),
        inputSize, size, batchSize, false, false);

    // Initialize output to same size and values for convenience.
    arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
        inputSize, size, batchSize, false, false);
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
    arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
        input.n_rows / size, size, batchSize, false, false);

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

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  g.set_size(arma::size(input));
  arma::cube gyTemp(const_cast<arma::Mat<eT>&>(gy).memptr(),
      input.n_rows / size, size, input.n_cols, false, false);
  arma::cube gTemp(const_cast<arma::Mat<eT>&>(g).memptr(),
      input.n_rows / size, size, input.n_cols, false, false);

  // Step 1: dl / dxhat.
  arma::cube norm = gyTemp.each_slice() % arma::repmat(gamma.t(),
      input.n_rows / size, 1);

  // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
  arma::mat temp = arma::sum(norm % inputMean, 2);
  arma::mat vars = temp % arma::repmat(arma::pow(stdInv, 3),
      input.n_rows / size, 1) * -0.5;

  // Step 3: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
  // dl / dmu * 1 / m.
  gTemp = (norm.each_slice() % arma::repmat(stdInv,
      input.n_rows / size, 1) +
      (inputMean.each_slice() % vars * 2)) / input.n_cols;

  // Step 4: sum (dl / dxhat * -1 / stdInv) + variance *
  // (sum -2 * (x - mu)) / m.
  arma::mat normTemp = arma::sum(norm.each_slice() %
      arma::repmat(-stdInv, input.n_rows / size, 1) , 2) /
      input.n_cols;
  gTemp.each_slice() += normTemp;
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient.set_size(size + size, 1);
  arma::cube errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(),
      error.n_rows / size, size, error.n_cols, false, false);

  // Step 5: dl / dy * xhat.
  arma::mat temp = arma::sum(arma::sum(normalized % errorTemp, 0), 2);
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = temp.t();

  // Step 6: dl / dy.
  temp = arma::sum(arma::sum(errorTemp, 0), 2);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = temp.t();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BatchNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(size);

  if (Archive::is_loading::value)
  {
    weights.set_size(size + size, 1);
    loading = false;
  }

  ar & CEREAL_NVP(eps);
  ar & CEREAL_NVP(gamma);
  ar & CEREAL_NVP(beta);
  ar & CEREAL_NVP(count);
  ar & CEREAL_NVP(averageFactor);
  ar & CEREAL_NVP(momentum);
  ar & CEREAL_NVP(average);
  ar & CEREAL_NVP(runningMean);
  ar & CEREAL_NVP(runningVariance);
}

} // namespace ann
} // namespace mlpack

#endif
