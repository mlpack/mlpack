/**
 * @file methods/ann/loss_functions/vr_class_reward.hpp
 * @author Marcus Edel
 *
 * Definition of the VRClassRewardType class, which implements the variance
 * reduced classification reinforcement layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LOSS_FUNCTIONS_VR_CLASS_REWARD_HPP
#define MLPACK_METHODS_ANN_LOSS_FUNCTIONS_VR_CLASS_REWARD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * Implementation of the variance reduced classification reinforcement layer.
 * This layer is meant to be used in combination with the reinforce normal layer
 * (ReinforceNormalLayer), which expects that the reward is 1 for success, and 0
 * otherwise.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class VRClassRewardType
{
 public:
  /**
   * Create the VRClassRewardType object.
   *
   * @param scale Parameter used to scale the reward.
   * @param sizeAverage Take the average over all batches.
   */
  VRClassRewardType(const double scale = 1, const bool sizeAverage = true);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data that contains the log-probabilities for each class.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  typename MatType::elem_type Forward(const MatType& input,
                                      const MatType& target);

  /**
   * Ordinary feed backward pass of a neural network. The negative log
   * likelihood layer expectes that the input contains log-probabilities for
   * each class. The layer also expects a class index, in the range between 1
   * and the number of classes, as target when calling the Forward function.
   *
   * @param input The propagated input activation.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   * @param output The calculated error.
   */
  void Backward(const MatType& input, const MatType& target, MatType& output);

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <typename LayerType, typename... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(Layer<MatType>* layer)
  {
    network.push_back(layer);
  }

  //! Get the network.
  const std::vector<Layer<MatType>*>& Network() const { return network; }
  //! Modify the network.
  std::vector<Layer<MatType>*>& Network() { return network; }

  //! Get the value of parameter sizeAverage.
  bool SizeAverage() const { return sizeAverage; }

  //! Get the value of scale parameter.
  double Scale() const { return scale; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */);

 private:
  //! Locally-stored value to scale the reward.
  double scale;

  //! If true take the average over all batches.
  bool sizeAverage;

  //! Locally stored reward parameter.
  double reward;

  //! Locally-stored network modules.
  std::vector<Layer<MatType>*> network;
}; // class VRClassRewardType

// Default typedef for typical `arma::mat` usage.
using VRClassReward = VRClassRewardType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "vr_class_reward_impl.hpp"

#endif
