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
    momentum(0.1),
    loading(false),
    deterministic(false),
    count(0)
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
    count(0)
{
  weights.set_size(size + size, 1);
  runningMean.zeros(size, 1);
  runningVariance.zeros(size, 1);
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

  // Set size of output equal to the size of input.
  output.set_size(arma::size(input));

  // We will calculate minibatch norm on each channel / feature map.
  if (!deterministic)
  {
    // Input size refers to inputWidth * inputHeight.
    // If inputSize equals one, input corresponds to output from linear layer.
    const size_t inputSize = input.n_rows / size;

    // Use Welford method to compute the sample variance and mean.
    for (size_t i = 0; i < input.n_cols; ++i)
    {
      mean = arma::mean(input, 1);
      variance = arma::var(input, 1, 1);

      // Normalize the input.
      output = input.each_col() - mean;
      inputMean = output;
      output.each_col() /= arma::sqrt(variance + eps);

      // Use Welford method to compute the sample variance and mean.
      for (size_t i = 0; i < input.n_cols; i++)
      {
        count += 1;

        OutputDataType diff = input.col(i) - runningMean;
        if (average)
        {
          runningMean = runningMean + diff / count;
          runningVariance += diff % (input.col(i) - runningMean);
        }
        else
        {
          runningMean = (1 - momentum) * runningMean + momentum * mean;
          runningVariance = (1 - momentum) * runningVariance +
              momentum * variance;
        }
      }

      // Reused in the backward and gradient step.
      normalized = output;

      // Scale and shift the output.
      output.each_col() %= gamma;
      output.each_col() += beta;
      return;
    }

    // Input corresponds to output from convolution layer.
    // Use a cube for simplicity.
    size_t batchSize = input.n_cols;
    arma::cube inputTemp(const_cast<arma::Mat<eT>&>(input).memptr(),
        input.n_rows / size, size, batchSize, false, false);

    // Initialize output to same size and values for convenience.
    arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);
    outputTemp = inputTemp;

    // Calculate mean and variance over all channels.
    mean = arma::mean(arma::mean(inputTemp, 2), 0);
    variance = arma::mean(arma::mean(arma::pow(
        inputTemp.each_slice() - arma::repmat(mean,
        input.n_rows / size, 1), 2), 2), 0);

    outputTemp.each_slice() -= arma::repmat(mean, input.n_rows / size, 1);

    // Used in backward propagation.
    inputMean.set_size(arma::size(input));

    arma::cube inputMeanTemp(const_cast<arma::Mat<eT>&>(inputMean).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);
    inputMeanTemp = outputTemp;

    // Normalize output.
    outputTemp.each_slice() /= arma::sqrt(arma::repmat(variance,
        input.n_rows / size, 1) + eps);

    // Re-used in backward propagation
    normalized.set_size(arma::size(input));
    arma::cube normalizedTemp(const_cast<arma::Mat<eT>&>(normalized).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);
    normalizedTemp = outputTemp;

    outputTemp.each_slice() %= arma::repmat(gamma.t(),
        input.n_rows / size, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(),
        input.n_rows / size, 1);

    if (!average)
    {
      double nElements = (double) input.n_elem / size;
      nElements = 1.0 * nElements / (1.0 * nElements - 1.0);

      runningMean = (1 - momentum) * runningMean + momentum * mean.t();
      runningVariance = (1 - momentum) * runningVariance +
          nElements * momentum * variance.t();
    }
    else
    {
      // Use Welford method to update running mean and variance.
      count += input.n_cols;
      arma::mat diff = arma::sum(arma::sum(inputTemp.each_slice() -
          arma::repmat(runningMean.t(), inputSize, 1), 2), 0) / count;
      runningMean += diff.t() / count;
      runningVariance += diff.t() % diff.t();
    }
  }
  else
  {
    // Normalize the input and scale and shift the output.
    if (input.n_rows == size)
    {
      output = input.each_col() - runningMean;
      output.each_col() %= gamma / arma::sqrt(runningVariance / count + eps);
      output.each_col() += beta;
    }
    else
    {
      output = input;
      arma::cube outputTemp(const_cast<arma::Mat<eT>&>(output).memptr(),
          input.n_rows / size, size, input.n_cols, false, false);

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
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);

  // Input corresponds to a linear region.
  if (size == input.n_rows)
  {
    // Step 1: dl / dxhat.
    const arma::mat norm = gy.each_col() % gamma;

    // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
    const arma::mat var = arma::sum(norm % inputMean, 1) %
        arma::pow(stdInv, 3.0) * -0.5;

    // Step 4: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
    // dl / dmu * 1 / m.
    g = (norm.each_col() % stdInv) + (inputMean.each_col() %
        var * 2 / input.n_cols);

    // Step 3: sum (dl / dxhat * -1 / stdInv) + variance *
    // (sum -2 * (x - mu)) / m.
    g.each_col() += arma::sum(norm.each_col() % -stdInv, 1) / input.n_cols;
  }
  else
  {
    g.set_size(arma::size(input));
    arma::cube gyTemp(const_cast<arma::Mat<eT>&>(gy).memptr(),
          input.n_rows / size, size, input.n_cols, false, false);
    arma::cube gTemp(const_cast<arma::Mat<eT>&>(g).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);
    arma::cube inputMeanTemp(const_cast<arma::Mat<eT>&>(inputMean).memptr(),
        input.n_rows / size, size, input.n_cols, false, false);

    // Step 1: dl / dxhat.
    arma::cube norm = gyTemp.each_slice() % arma::repmat(gamma.t(),
          input.n_rows / size, 1);

    // Step 2: sum dl / dxhat * (x - mu) * -0.5 * stdInv^3.
    arma::mat temp = arma::sum(norm % inputMeanTemp, 2);
    arma::mat vars = temp % arma::repmat(arma::pow(stdInv, 3),
        input.n_rows / size, 1) * -0.5;

    // Step 3: dl / dxhat * 1 / stdInv + variance * 2 * (x - mu) / m +
    // dl / dmu * 1 / m.
    gTemp = (norm.each_slice() % arma::repmat(stdInv,
        input.n_rows / size, 1) +
        (inputMeanTemp.each_slice() % vars * 2)) / input.n_cols;

    // Step 4: sum (dl / dxhat * -1 / stdInv) + variance *
    // (sum -2 * (x - mu)) / m.
    arma::mat normTemp = arma::sum(norm.each_slice() %
        arma::repmat(-stdInv, input.n_rows / size, 1) , 2) /
        input.n_cols;
    gTemp =  gTemp.each_slice() + normTemp;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void BatchNorm<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& /* input */,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& gradient)
{
  gradient.set_size(size + size, 1);

  if (error.n_rows == size)
  {
    // Step 5: dl / dy * xhat.
    gradient.submat(0, 0, gamma.n_elem - 1, 0) = arma::sum(normalized %
        error, 1);

    // Step 6: dl / dy.
    gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) =
        arma::sum(error, 1);
    return;
  }

  arma::cube normalizedTemp(const_cast<arma::Mat<eT>&>(normalized).memptr(),
        error.n_rows / size, size, error.n_cols, false, false);
  arma::cube errorTemp(const_cast<arma::Mat<eT>&>(error).memptr(),
      error.n_rows / size, size, error.n_cols, false, false);

  // Step 5: dl / dy * xhat.
  arma::mat temp = arma::sum(arma::sum(normalizedTemp % errorTemp, 2), 0);
  gradient.submat(0, 0, gamma.n_elem - 1, 0) = temp.t();

  // Step 6: dl / dy.
  temp = arma::sum(arma::sum(errorTemp, 2), 0);
  gradient.submat(gamma.n_elem, 0, gradient.n_elem - 1, 0) = temp.t();
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void BatchNorm<InputDataType, OutputDataType>::serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(size);

  if (Archive::is_loading::value)
  {
    weights.set_size(size + size, 1);
    loading = false;
  }

  ar & BOOST_SERIALIZATION_NVP(eps);
  ar & BOOST_SERIALIZATION_NVP(gamma);
  ar & BOOST_SERIALIZATION_NVP(beta);
  ar & BOOST_SERIALIZATION_NVP(count);
  ar & BOOST_SERIALIZATION_NVP(momentum);
  ar & BOOST_SERIALIZATION_NVP(average);
  ar & BOOST_SERIALIZATION_NVP(runningMean);
  ar & BOOST_SERIALIZATION_NVP(runningVariance);
}

} // namespace ann
} // namespace mlpack

#endif
