/**
 * @file methods/ann/layer/virtual_batch_norm.hpp
 * @author Saksham Bansal
 *
 * Definition of the VirtualBatchNorm layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_VIRTUALBATCHNORM_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

// TODO: what about sizes for this layer?
/**
 * Declaration of the VirtualBatchNorm layer class. Instead of using the
 * batch statistics for normalizing on a mini-batch, it uses a reference subset
 * of the data for calculating the normalization statistics.
 *
 * For more information, refer to the following paper,
 *
 * @code
 * @article{Goodfellow2016,
 *   author  = {Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung,
 *              Alec Radford, Xi Chen},
 *   title   = {Improved Techniques for Training GANs},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1606.03498},
 * }
 * @endcode
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputType = arma::mat,
  typename OutputType = arma::mat
>
class VirtualBatchNormType : public Layer<InputType, OutputType>
{
 public:
  //! Create the VirtualBatchNorm object.
  VirtualBatchNormType();

  /**
   * Create the VirtualBatchNorm layer object for a specified number of input
   * units.
   *
   * @param referenceBatch The data from which the normalization
   *        statistics are computed.
   * @param size The number of input units / channels.
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  VirtualBatchNormType(const InputType& referenceBatch,
                       const size_t size,
                       const double eps = 1e-8);

  //! Clone the VirtualBatchNormType object. This handles polymorphism
  //! correctly.
  VirtualBatchNormType* Clone() const
  {
    return new VirtualBatchNormType(*this);
  }

  /**
   * Reset the layer parameters.
   */
  void SetWeights(typename OutputType::elem_type* weightsPtr);

  /**
   * Forward pass of the Virtual Batch Normalization layer. Transforms the input
   * data into zero mean and unit variance, scales the data by a factor gamma
   * and shifts it by beta.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Backward pass through the layer.
   *
   * @param * (input) The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param * (input) The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& /* input */,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the number of input units.
  size_t InSize() const { return size; }

  //! Get the epsilon value.
  double Epsilon() const { return eps; }

  const size_t WeightSize() const
  {
    return 2 * size;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored number of input units.
  size_t size;

  //! Locally-stored epsilon value.
  double eps;

  //! Variable to keep track of whether we are in loading or saving mode.
  bool loading;

  //! Locally-stored scale parameter.
  OutputType gamma;

  //! Locally-stored shift parameter.
  OutputType beta;

  //! Locally-stored parameters.
  OutputType weights;

  //! Mean of features in the reference batch.
  OutputType referenceBatchMean;

  //! Variance of features in the reference batch.
  OutputType referenceBatchMeanSquared;

  //! The coefficient for reference batch statistics.
  double oldCoefficient;

  //! The coefficient for input batch statistics.
  double newCoefficient;

  //! Locally-stored mean object.
  OutputType mean;

  //! Locally-stored variance object.
  OutputType variance;

  //! Locally-stored normalized input.
  OutputType normalized;

  //! Locally-stored zero mean input.
  OutputType inputSubMean;
}; // class VirtualBatchNormType

// Standard VirtualBatchNorm layer.
using VirtualBatchNorm = VirtualBatchNormType<arma::mat, arma::mat>;

} // namespace mlpack

// Include the implementation.
#include "virtual_batch_norm_impl.hpp"

#endif
