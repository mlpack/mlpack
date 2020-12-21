/**
 * @file methods/ann/layer/layer.hpp
 * @author Marcus Edel
 *
 * This includes various layers to construct a model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_LAYER_HPP

template<typename InputDataType, typename OutputDataType>
class Layer
{
 public:
  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  virtual void Forward(const arma::mat&,
                       arma::mat&)
  { /* Nothing to do here */ }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  virtual double Forward(const arma::mat&,
                         const arma::mat&)
  { /* Nothing to do here */ }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  virtual void Backward(const arma::mat& input,
                        const arma::mat& gy,
                        arma::mat& g)
  { /* Nothing to do here */ }

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  virtual void Gradient(const arma::mat& input,
                        const arma::mat& error,
                        arma::mat& gradient)
  { /* Nothing to do here */ }

  /**
   * Reset the layer parameter.
   */
  virtual void Reset() {}

  //! Get the gradient.
  virtual OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  virtual OutputDataType& Gradient() { return gradient; }

  //! Get the delta.
  virtual arma::mat const& Delta() const { return delta; }
  //! Modify the delta.
  virtual arma::mat& Delta() { return delta; }

  //! Get the layer loss.
  virtual double Loss() { return 0; }

  //! Get the size of the weights.
  virtual size_t WeightSize() const { return 0; }

  //! Get the parameters.
  virtual OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  virtual OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  virtual InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  virtual InputDataType& InputParameter() { return inputParameter; }

  virtual OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  virtual OutputDataType& OutputParameter() { return outputParameter; }

 private:
  arma::mat weights;

  arma::mat outputParameter;

  arma::mat inputParameter;

  arma::mat delta;

  arma::mat gradient;
};

#endif
