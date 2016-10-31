/**
 * @file vr_class_reward_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the VRClassRewardLayer class, which implements the variance
 * reduced classification reinforcement layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the variance reduced classification reinforcement layer.
 * This layer is meant to be used in combination with the reinforce normal layer
 * (ReinforceNormalLayer), which expects that an reward:
 * (1 for success, 0 otherwise).
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::field<arma::mat>,
    typename OutputDataType = arma::field<arma::mat>
>
class VRClassRewardLayer
{
 public:
  /**
   * Create the VRClassRewardLayer object.
   *
   * @param scale Parameter used to scale the reward.
   * @param sizeAverage Take the average over all batches.
   */
  VRClassRewardLayer(const double scale = 1, const bool sizeAverage = true) :
      scale(scale),
      sizeAverage(sizeAverage)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data that contains the log-probabilities for each class.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename eT>
  double Forward(const arma::field<arma::Mat<eT> >& input,
                 const arma::Mat<eT>& target)
  {
    return Forward(input(0, 0), target);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data that contains the log-probabilities for each class.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>& input, const arma::Mat<eT>& target)
  {
    reward = 0;
    arma::uword index = 0;

    for (size_t i = 0; i < input.n_cols; i++)
    {
      input.unsafe_col(i).max(index);
      reward = ((index + 1) == target(i)) * scale;
    }

    if (sizeAverage)
    {
      return -reward / input.n_cols;
    }

    return -reward;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  double Backward(const arma::field<arma::Mat<eT> >& input,
                const arma::Mat<eT>& /* gy */,
                arma::field<arma::Mat<eT> >& g)
  {
    g = arma::field<arma::Mat<eT> >(2, 1);
    g(0, 0) = arma::zeros(input(0, 0).n_rows, input(0, 0).n_cols);

    double vrReward = reward - arma::as_scalar(input(1, 0));
    if (sizeAverage)
    {
      vrReward /= input(0, 0).n_cols;
    }

    const double norm = sizeAverage ? 2.0 / input.n_cols : 2.0;

    g(1, 0) = norm * (input(1, 0) - reward);

    return vrReward;
  }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

 private:
  //! Locally-stored value to scale the reward.
  const double scale;

  //! If true take the average over all batches.
  const bool sizeAverage;

  //! Locally stored reward parameter.
  double reward;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;
}; // class VRClassRewardLayer

}; // namespace ann
}; // namespace mlpack

#endif
