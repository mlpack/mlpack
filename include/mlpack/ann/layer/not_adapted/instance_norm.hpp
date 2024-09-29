/**
 * @file methods/ann/layer/instance_norm.hpp
 * @author Anjishnu Mukherjee
 * @author Shah Anwaar Khalid
 *
 * Definition of the Instance Normalization layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_INSTANCE_NORM_HPP
#define MLPACK_METHODS_ANN_LAYER_INSTANCE_NORM_HPP

#include <mlpack/prereqs.hpp>
#include "layer_types.hpp"

namespace mlpack {

/**
 * Declaration of the Instance Normalization layer class. The layer transforms
 * the input data into zero mean and unit variance and then scales and shifts
 * the data by parameters, gamma and beta respectively. These parameters are
 * learnt by the network. The mean and standard-deviation are calculated
 * per-dimension separately for each object in a mini-batch.
 *
 * If deterministic is false (training), the mean and variance are calculated
 * and the data is normalized. If it is set to true (testing) then
 * the mean and variance accrued over the training set is used.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Ulyanov17,
 *   author    = {Dmitry Ulyanov, Andrea Vedaldi and
 *                Victor Lempitsky},
 *   title     = {Instance Normalization:
 *                The Missing Ingredient for Fast Stylization},
 *   year      = {2017},
 *   url       = {https://arxiv.org/abs/1607.08022}
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
class InstanceNorm
{
 public:
  //! Create the InstanceNorm object.
  InstanceNorm();

  /**
   * Create the InstanceNorm layer object with the specified parameters.
   *
   * @param size The number of input units / channels.
   * @param batchSize Size of the minibatch.
   * @param eps The epsilon added to variance to ensure numerical stability.
   * @param average Boolean to determine whether cumulative average is used for
   *                updating the parameters or momentum.
   * @param momentum Parameter used to update the running mean and variance.
   */
  InstanceNorm(const size_t size,
               const size_t batchSize,
               const double eps = 1e-5,
               const bool average = true,
               const double momentum = 0.1);

  /**
   * Forward pass of the Instance Normalization layer. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer
   * @param output Resulting output activations.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Backward pass through the layer.
   *
   * @param input The input activations
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations
   * @param error The calculated error
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return batchNorm.Parameters(); }
  //! Modify the parameters.
  OutputDataType& Parameters() { return batchNorm.Parameters(); }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const
  { return batchNorm.OutputParameter(); }

  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return batchNorm.OutputParameter(); }

  //! Get the delta.
  OutputDataType const& Delta() const { return batchNorm.Delta(); }
  //! Modify the delta.
  OutputDataType& Delta() { return batchNorm.Delta(); }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return batchNorm.Gradient(); }
  //! Modify the gradient.
  OutputDataType& Gradient() { return batchNorm.Gradient(); }

  //! Get the value of deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! Get the mean over the training data.
  OutputDataType const& TrainingMean() const { return runningMean; }
  //! Modify the mean over the training data.
  OutputDataType& TrainingMean() { return runningMean; }

  //! Get the variance over the training data.
  OutputDataType const& TrainingVariance() const { return runningVariance; }
  //! Modify the variance over the training data.
  OutputDataType& TrainingVariance() { return runningVariance; }

  //! Get the number of input units / channels.
  size_t InputSize() const { return size; }
  //! Modify the input units/ channels.
  size_t InputSize() {return size; }

  //! Get the epsilon value.
  double Epsilon() const { return eps; }
  //! Modify the epsilon value.
  double Epsilon() { return eps; }


  //! Get the momentum value.
  double Momentum() const { return momentum; }
  //! Modify the momentum value.
  double Momentum() { return momentum; }

  //! Get the average parameter.
  bool Average() const { return average; }
  //! Modify the average parameter.
  bool Average() { return average; }

  //! Get the batchSize parameter.
  bool Batchsize() const { return batchSize; }
  //! Modify the batchSize parameter.
  bool Batchsize() { return batchSize; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally stored BatchNorm Object.
  BatchNorm<InputDataType, OutputDataType> batchNorm;

  //! Locally-stored reset parameter used to initialize the layer once.
  bool reset;

  //! Locally-stored number of input units.
  size_t size;

  //! Locally-stored epsilon value.
  double eps;

  //! If true use average else use momentum for computing running mean
  //! and variance
  bool average;

  //! Locally-stored value for momentum.
  double momentum;

  //! Locally stored vale for numFunctions
  size_t batchSize;

  /**
   * If true then mean and variance over the training set will be considered
   * instead of being calculated over the batch.
   */
  bool deterministic;

  //! Locally-stored mean object.
  OutputDataType runningMean;

  //! Locally-stored variance object.
  OutputDataType runningVariance;
}; // class InstanceNorm

} // namespace mlpack

// Include the implementation.
#include "instance_norm_impl.hpp"

#endif
