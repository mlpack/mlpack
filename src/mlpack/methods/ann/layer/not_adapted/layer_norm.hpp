/**
 * @file methods/ann/layer/layer_norm.hpp
 * @author Shikhar Jaiswal
 *
 * Definition of the Layer Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYERNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYERNORM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Declaration of the Layer Normalization class. The layer transforms
 * the input data into zero mean and unit variance and then scales and shifts
 * the data by parameters, gamma and beta respectively over a single training
 * data. These parameters are learnt by the network. Layer Normalization is
 * different from Batch Normalization in the way that normalization is done
 * for individual training cases, and the mean and standard deviations are
 * computed across the layer dimensions, as opposed to across the batch.
 *
 * For more information, refer to the following papers,
 *
 * @code
 * @article{Ba16,
 *   author    = {Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey E. Hinton},
 *   title     = {Layer Normalization},
 *   volume    = {abs/1607.06450},
 *   year      = {2016},
 *   url       = {http://arxiv.org/abs/1607.06450},
 *   eprint    = {1607.06450},
 * }
 * @endcode
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
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputType = arma::mat,
  typename OutputType = arma::mat
>
class LayerNormType : public Layer<InputType, OutputType>
{
 public:
  //! Create the LayerNormType object.
  LayerNormType();

  /**
   * Create the LayerNorm object for a specified number of input units.
   *
   * @param size The number of input units.
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  LayerNormType(const size_t size, const double eps = 1e-8);

  //! Clone the LayerNormType object. This handles polymorphism correctly.
  LayerNormType* Clone() const { return new LayerNormType(*this); }

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Forward pass of Layer Normalization. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Backward pass through the layer.
   *
   * @param input The input activations.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& input,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the mean across single training data.
  OutputType Mean() { return mean; }

  //! Get the variance across single training data.
  OutputType Variance() { return variance; }

  //! Get the number of input units.
  size_t InSize() const { return size; }

  //! Get the value of epsilon.
  double Epsilon() const { return eps; }

  const size_t WeightSize() const { return 2 * size; }

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

  //! Locally-stored mean object.
  OutputType mean;

  //! Locally-stored variance object.
  OutputType variance;

  //! Locally-stored normalized input.
  OutputType normalized;

  //! Locally-stored zero mean input.
  OutputType inputMean;
}; // class LayerNormType

// Standard LayerNorm type
typedef LayerNormType<arma::mat, arma::mat> LayerNorm;

} // namespace mlpack

// Include the implementation.
#include "layer_norm_impl.hpp"

#endif
