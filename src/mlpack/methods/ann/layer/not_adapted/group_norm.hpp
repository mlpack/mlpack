/**
 * @file methods/ann/layer/group_norm.hpp
 * @author Abhinav Anand
 *
 * Definition of the Group Normalization class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GROUPNORM_HPP
#define MLPACK_METHODS_ANN_LAYER_GROUPNORM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Declaration of the Group Normalization class. The group transforms
 * the input data into zero mean and unit variance and then scales and shifts
 * the data by parameters, gamma and beta respectively over a single training
 * data. These parameters are learnt by the network. Group Normalization is
 * different from Group Normalization in the way that normalization is done
 * for individual training cases, and the mean and standard deviations are
 * computed across the layer dimensions divided into groups,
 * as opposed to across the group.
 *
 * For more information, refer to the following papers,
 *
 * @code
 * @article{wu2018group,
 *   author    = {Wu, Yuxin and He, Kaiming},
 *   title     = {Group normalization},
 *   year      = {2018},
 *   url       = {https://arxiv.org/abs/1803.08494}
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
class GroupNorm
{
 public:
  //! Create the GroupNorm object.
  GroupNorm();

  /**
   * Create the GroupNorm object for a specified number of input units.
   *
   * @param size The number of input units.
   * @param eps The epsilon added to variance to ensure numerical stability.
   */
  GroupNorm(const size_t groupCount,
            const size_t size,
            const double eps = 1e-8);

  /**
   * Reset the layer parameters.
   */
  void Reset();

  /**
   * Forward pass of Group Normalization. Transforms the input data
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
   * @param input The input activations.
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
   * @param input The input activations.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& input,
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

  //! Get the mean across single training data.
  OutputDataType Mean() { return mean; }

  //! Get the variance across single training data.
  OutputDataType Variance() { return variance; }

  //! Get the number of input units.
  size_t InSize() const { return size; }

  //! Get the value of epsilon.
  double Epsilon() const { return eps; }

  //! Get the shape of the input.
  size_t InputShape() const
  {
    return size;
  }

  //! Get the group count.
  size_t GroupCount() const
  {
    return groupCount;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored group count.
  size_t groupCount;

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

  //! Locally-stored normalized input.
  OutputDataType normalized;

  //! Locally-stored temp matrix.
  OutputDataType temp;

  //! Locally-stored zero mean input.
  OutputDataType inputMean;
}; // class GroupNorm

} // namespace mlpack

// Include the implementation.
#include "group_norm_impl.hpp"

#endif
