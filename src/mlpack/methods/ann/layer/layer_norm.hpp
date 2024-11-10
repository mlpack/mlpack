/**
 * @file methods/ann/layer/layer_norm.hpp
 * @author Shikhar Jaiswal
 * @author Adam Kropp
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
 * @tparam MatType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename MatType = arma::mat
>
class LayerNormType : public Layer<MatType>
{
 public:
  /**
   * Create the LayerNorm object for a specified number of input units.
   *
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  LayerNormType(const double eps = 1e-8);

  //! Clone the LayerNormType object. This handles polymorphism correctly.
  LayerNormType* Clone() const override { return new LayerNormType(*this); }

  /**
   * Forward pass of Layer Normalization. Transforms the input data
   * into zero mean and unit variance, scales the data by a factor gamma and
   * shifts it by beta.
   *
   * @param input Input data for the layer.
   * @param output Resulting output activations.
   */
  void Forward(const MatType& input, MatType& output) override;

  /**
   * Backward pass through the layer.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& /* output */,
                const MatType& gy,
                MatType& g) override;

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& error,
                MatType& gradient) override;

  //! Get the parameters.
  MatType const& Parameters() const override { return weights; }
  //! Modify the parameters.
  MatType& Parameters() override { return weights; }

  //! Get the mean across single training data.
  MatType Mean() { return mean; }

  //! Get the variance across single training data.
  MatType Variance() { return variance; }

  //! Get the number of input units.
  size_t InSize() const { return size; }

  //! Get the value of epsilon.
  double Epsilon() const { return eps; }

  size_t WeightSize() const override { return 2 * size; }

  void ComputeOutputDimensions() override
  {
    // The default implementation is to assume that the output size is the same
    // as the input.
    this->outputDimensions = this->inputDimensions;
    size = this->inputDimensions[0];
    for (size_t i = 1; i < this->inputDimensions.size(); i++)
      size *= this->inputDimensions[i];
  }

  void SetWeights(const MatType& weightsIn) override;

  void CustomInitialize(
      MatType& /* W */,
      const size_t /* elements */) override;

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored epsilon value.
  double eps;

  // Cached size for the normalization.
  size_t size;

  //! Locally-stored scale parameter.
  MatType gamma;

  //! Locally-stored shift parameter.
  MatType beta;

  //! Locally-stored parameters.
  MatType weights;

  //! Locally-stored mean object.
  MatType mean;

  //! Locally-stored variance object.
  MatType variance;

  //! Locally-stored normalized input.
  MatType normalized;

  //! Locally-stored zero mean input.
  MatType inputMean;
}; // class LayerNormType

// Standard LayerNorm type
using LayerNorm = LayerNormType<arma::mat>;

} // namespace mlpack

// Include the implementation.
#include "layer_norm_impl.hpp"

#endif
