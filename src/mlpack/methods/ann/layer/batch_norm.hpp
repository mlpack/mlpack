/**
 * @file batch_norm.hpp
 * @author Praveen Ch
 * @author Manthan-R-Sheth
 *
 * Definition of the Batch Normalization layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_BATCHNORM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Declaration of the Batch Normalization layer class. The layer transforms
 * the input data into zero mean and unit variance and then scales and shifts
 * the data by parameters, gamma and beta respectively. These parameters are
 * learnt by the network.
 *
 * If deterministic is false (training), the mean and variance over the batch is
 * calculated and the data is normalized. If it is set to true (testing) then
 * the mean and variance accrued over the training set is used.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Ioffe15,
 *   author    = {Sergey Ioffe and
 *                Christian Szegedy},
 *   title     = {Batch Normalization: Accelerating Deep Network Training by
 *                Reducing Internal Covariate Shift},
 *   journal   = {CoRR},
 *   volume    = {abs/1502.03167},
 *   year      = {2015},
 *   url       = {http://arxiv.org/abs/1502.03167},
 *   eprint    = {1502.03167},
 * }
 * @endcode
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
>
class BatchNorm
{
 public:
  //! Create the BatchNorm object.
  BatchNorm();

  /**
   * Create the BatchNorm layer object for a specified number of input units.
   *
   * @param size The number of input units.
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  BatchNorm(const size_t size, const double eps = 1e-8);

  /**
   * Reset the layer parameters
   */
  void Reset();

  /**
   * Forward pass of the Batch Normalization layer. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer
   * @param output Resulting output activations.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

  /**
   * Backward pass through the layer.
   *
   * @param input The input activations
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations
   * @param error The calculated error
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
                arma::Mat<eT>&& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! Get the value of deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the mean over the training data.
  OutputDataType TrainingMean() { return runningMean; }

  //! Get the variance over the training data.
  OutputDataType TrainingVariance() { return runningVariance / count; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored number of input units.
  size_t size;

  //! Locally-stored epsilon value.
  double eps;

  //! Variable to keep track of whether we are in loading or saving mode.
  bool loading;

  //! Locally-stored scale parameter.
  OutputDataType gamma;

  //! Locally-stored shift parameter.
  OutputDataType beta;

  //! Locally-stored parameters.
  OutputDataType weights;

  /**
   * If true then mean and variance over the training set will be considered
   * instead of being calculated over the batch.
   */
  bool deterministic;

  //! Locally-stored running mean/variance counter.
  size_t count;

  //! Locally-stored mean object.
  OutputDataType mean;

  //! Locally-stored variance object.
  OutputDataType variance;

  //! Locally-stored mean object.
  OutputDataType runningMean;

  //! Locally-stored variance object.
  OutputDataType runningVariance;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored normalized input.
  OutputDataType normalized;

  //! Locally-stored zero mean input.
  OutputDataType inputMean;
}; // class BatchNorm

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "batch_norm_impl.hpp"

#endif
