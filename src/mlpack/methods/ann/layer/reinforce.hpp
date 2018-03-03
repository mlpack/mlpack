/**
 * @file reinforce.hpp
 * @author Shangtong Zhang
 *
 * Definition of the Reinforce class, which implements the REINFORCE algorithm.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_HPP
#define MLPACK_METHODS_ANN_LAYER_REINFORCE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Reinforce layer, which is meant to be the output layer
 * for the policy gradient algorithm. When used as the output layer, the last
 * layer of the network should be the Policy layer.
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
class Reinforce
{
 public:
  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  double Forward(const arma::Mat<eT>&&, arma::Mat<eT>&& );

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType&& input, DataType&& /* gy */, DataType&& g);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */);

 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //!  Locally-stored output module parameter parameters.
  std::vector<arma::mat> moduleInputParameter;
}; // class Reinforce

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reinforce_impl.hpp"

#endif
