/**
 * @file methods/ann/layer/batch_norm.hpp
 * @author Praveen Ch
 * @author Manthan-R-Sheth
 * @author Shubham Agrawal
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

#include "layer.hpp"

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
 * @tparam MatType Matrix representation to accept as input and use for
 *         computation.
 */
template <typename MatType = arma::mat>
class BatchNormType : public Layer<MatType>
{
 public:
  //! Create the BatchNorm object.
  BatchNormType();

  /**
   * Create the BatchNorm layer object for a specified number of input units.
   *
   * @param maxAxis The max axis along which BatchNorm is applied. After that,
   *                it will be treated as another higher dimension point.
   * @param eps The epsilon added to variance to ensure numerical stability.
   * @param average Boolean to determine whether cumulative average is used for
   *                updating the parameters or momentum is used.
   * @param momentum Parameter used to to update the running mean and variance.
   */
  BatchNormType(const size_t maxAxis,
                const double eps = 1e-8,
                const bool average = true,
                const double momentum = 0.1);
  
  virtual ~BatchNormType() { }

  //! Clone the BatchNormType object. This handles polymorphism correctly.
  BatchNormType* Clone() const { return new BatchNormType(*this); }

  //! Copy the other BatchNorm layer (but not weights).
  BatchNormType(const BatchNormType& layer);

  //! Take ownership of the members of the other BatchNorm layer (but not weights).
  BatchNormType(BatchNormType&& layer);

  //! Copy the other BatchNorm layer (but not weights).
  BatchNormType& operator=(const BatchNormType& layer);

  //! Take ownership of the members of the other BatchNorm layer (but not weights).
  BatchNormType& operator=(BatchNormType&& layer);

  /**
   * Reset the layer parameters.
   */
  void SetWeights(typename MatType::elem_type* weightsPtr);

  /**
   * Initialize the weight matrix of the layer.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param * (cols) Number of columns.
   */
  void CustomInitialize(
    MatType& W,
    const size_t rows, 
    const size_t /* cols */);

  /**
   * Forward pass of the Batch Normalization layer. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer
   * @param output Resulting output activations.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Backward pass through the layer.
   *
   * @param input The input activations
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input,
                const MatType& gy,
                MatType& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations
   * @param error The calculated error
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient);

  //! Get the parameters.
  const MatType& Parameters() const { return weights; }
  //! Modify the parameters.
  MatType& Parameters() { return weights; }

  //! Get the mean over the training data.
  const MatType& TrainingMean() const { return runningMean; }
  //! Modify the mean over the training data.
  MatType& TrainingMean() { return runningMean; }

  //! Get the variance over the training data.
  const MatType& TrainingVariance() const { return runningVariance; }
  //! Modify the variance over the training data.
  MatType& TrainingVariance() { return runningVariance; }

  //! Get the number of input units / channels.
  size_t InputSize() const { return size; }

  //! Get the epsilon value.
  double Epsilon() const { return eps; }

  //! Get the momentum value.
  double Momentum() const { return momentum; }

  //! Get the average parameter.
  bool Average() const { return average; }

  //! Get size of weights.
  size_t WeightSize() const { return 2 * size; }

  //! Compute the output dimensions of the layer given `InputDimensions()`.
  void ComputeOutputDimensions();

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored maxAxis along which BatchNorm will apply.
  size_t maxAxis;

  //! Locally-stored epsilon value.
  double eps;

  //! If true use average else use momentum for computing running mean
  //! and variance
  bool average;

  //! Locally-stored value for momentum.
  double momentum;

  //! Locally-stored scale parameter.
  MatType gamma;

  //! Locally-stored shift parameter.
  MatType beta;

  //! Locally-stored mean object.
  MatType mean;

  //! Locally-stored variance object.
  MatType variance;

  //! Locally-stored parameters.
  MatType weights;

  //! Locally-stored running mean/variance counter.
  size_t count;

  //! Locally-stored value for average factor which used to update running
  //! mean and variance.
  double averageFactor;

  //! Locally-stored input dimensions.
  size_t inputDimensions;

  //! Locally-stored number of input units.
  size_t size;

  //! Locally-stored higher dimension.
  size_t higherDimension;

  //! Locally-stored mean object.
  MatType runningMean;

  //! Locally-stored variance object.
  MatType runningVariance;

  //! Locally-stored normalized input.
  arma::Cube<typename MatType::elem_type> normalized;

  //! Locally-stored zero mean input.
  arma::Cube<typename MatType::elem_type> inputMean;
}; // class BatchNorm

// Convenience typedefs.

// Standard Adaptive max pooling layer.
typedef BatchNormType<arma::mat> BatchNorm;

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "batch_norm_impl.hpp"

#endif
