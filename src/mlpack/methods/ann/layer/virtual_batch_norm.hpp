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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Declaration of the VirtualBatchNorm layer class. Instead of using the
 * batch statistics for normalizing on a mini-batch, it uses a reference subset of
 * the data for calculating the normalization statistics.
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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
>
class VirtualBatchNorm
{
 public:
  //! Create the VirtualBatchNorm object.
  VirtualBatchNorm();

  /**
   * Create the VirtualBatchNorm layer object for a specified number of input units.
   *
   * @param referenceBatch The data from which the normalization
   *        statistics are computed.
   * @param size The number of input units / channels.
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  template<typename eT>
  VirtualBatchNorm(const arma::Mat<eT>& referenceBatch,
                   const size_t size,
                   const double eps = 1e-8);

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Forward pass of the Virtual Batch Normalization layer. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Backward pass through the layer.
   *
   * @param * (input) The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param * (input) The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& error,
                arma::Mat<eT>& gradient);

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

  //! Get the number of input units.
  size_t InSize() const { return size; }

  //! Get the epsilon value.
  double Epsilon() const { return eps; }

  /**
   * Serialize the layer.
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

  //! Mean of features in the reference batch.
  OutputDataType referenceBatchMean;

  //! Variance of features in the reference batch.
  OutputDataType referenceBatchMeanSquared;

  //! The coefficient for reference batch statistics.
  double oldCoefficient;

  //! The coefficient for input batch statistics.
  double newCoefficient;

  //! Locally-stored mean object.
  OutputDataType mean;

  //! Locally-stored variance object.
  OutputDataType variance;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored input parameter object.
  OutputDataType inputParameter;

  //! Locally-stored normalized input.
  OutputDataType normalized;

  //! Locally-stored zero mean input.
  OutputDataType inputSubMean;
}; // class VirtualBatchNorm

} // namespace ann
} // namespace mlpack

// Include the implementation.
#include "virtual_batch_norm_impl.hpp"

#endif
