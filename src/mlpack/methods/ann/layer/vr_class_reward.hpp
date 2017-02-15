/**
 * @file vr_class_reward.hpp
 * @author Marcus Edel
 *
 * Definition of the VRClassReward class, which implements the variance
 * reduced classification reinforcement layer.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_HPP
#define MLPACK_METHODS_ANN_LAYER_VR_CLASS_REWARD_HPP

#include <mlpack/prereqs.hpp>

#include "layer_types.hpp"

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
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class VRClassReward
{
 public:
  /**
   * Create the VRClassReward object.
   *
   * @param scale Parameter used to scale the reward.
   * @param sizeAverage Take the average over all batches.
   */
  VRClassReward(const double scale = 1, const bool sizeAverage = true);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data that contains the log-probabilities for each class.
   * @param target The target vector, that contains the class index in the range
   *        between 1 and the number of classes.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>&& input, const arma::Mat<eT>&& target);

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
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                const arma::Mat<eT>&& target,
                arma::Mat<eT>&& output);

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

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes layer) { network.push_back(layer); }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */);

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

  //! Locally-stored network modules.
  std::vector<LayerTypes> network;
}; // class VRClassReward

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "vr_class_reward_impl.hpp"

#endif
