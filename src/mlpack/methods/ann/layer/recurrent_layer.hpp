/**
 * @file methods/ann/layer/recurrent_layer.hpp
 * @author Ryan Curtin
 *
 * Base layer for recurrent neural network layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with the mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include "layer.hpp"

namespace mlpack {

/**
 * The `RecurrentLayer` provides a base layer for all layers that have recurrent
 * functionality and store state between steps in a recurrent network.  Any
 * RecurrentLayer should only be used with a network type such as `RNN` that
 * supports recurrent layers.
 *
 * Any recurrent layer that inherits from `RecurrentLayer` must implement the
 * `RecurrentSize()` function, which represents the number of elements that must
 * be held to represent the recurrent state for one data point and one time
 * step.  Recurrent state can be recovered with the `GetRecurrentState()`
 * function.  During BPTT, the gradients for recurrent connections are stored
 * and recovered with `RecurrentGradient()`.  More information on both of those
 * functions can be found in those functions' documentation.
 *
 * In addition, there are additional expectations for the Forward(), Backward()
 * and Gradient() methods:
 *
 * - `Forward()` should set the recurrent state for the current time step if
 *   `AtFinalStep()` is `false`; this can be done via a call to
 *   `GetRecurrentState(CurrentStep())`.
 *   - It is okay to set the recurrent state even if `AtFinalStep()` is `true`.
 *
 * - `Backward()` should use `GetRecurrentState(PreviousStep())` to refer to the
 *   recurrent input.  If `AtFinalStep()` is false,
 *   `GetRecurrentGradient(CurrentStep())`---which is the gradient of the
 *   network passed through the recurrent connection---must be considered
 *   (otherwise it can be assumed to be zero).  If `HasPreviousStep()` is
 *   true, then `Backward()` must also compute and store
 *   `GetRecurrentGradient(PreviousStep())`.
 *
 * - The same conditions that apply to `Backward()` also apply to `Gradient()`,
 *   but there is no need to compute and store
 *   `GetRecurrentGradient(PreviousStep())`.
 *
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class RecurrentLayer : public Layer<MatType>
{
 public:
  /**
   * Create the RecurrentLayer.
   */
  RecurrentLayer();

  // Virtual destructor is required for classes using inheritance.
  virtual ~RecurrentLayer() { }

  // Copy the given RecurrentLayer.
  RecurrentLayer(const RecurrentLayer& other);
  // Take ownership of the given RecurrentLayer.
  RecurrentLayer(RecurrentLayer&& other);
  // Copy the given RecurrentLayer.
  RecurrentLayer& operator=(const RecurrentLayer& other);
  // Take ownership of the given RecurrentLayer.
  RecurrentLayer& operator=(RecurrentLayer&& other);

  /**
   * ClearRecurrentState() is called before any forward pass of a recurrent
   * network.  This function is responsible for allocating any memory necessary
   * to store `bpttSteps` steps of previous forward and backward passes, with a
   * batch size of `batchSize`.  If `bpttSteps` is 0, then no space will be
   * initialized for the backward pass derivative state, and space for only the
   * previous timestep of the forward pass will be allocated.
   *
   * Any internal state of the recurrent layer will be set to 0.
   */
  void ClearRecurrentState(const size_t bpttSteps, const size_t batchSize);

  /**
   * Get the number of recurrent elements that need to be stored for a time
   * step.  A child recurrent class should override this.
   */
  virtual size_t RecurrentSize() const { return 0; }

  /**
   * Get the stored recurrent state at the given time step `t`.  If `t` is
   * greater than `CurrentStep()`, or if `t` is more than `bpttSteps` behind
   * `currentStep`, invalid results will be returned!
   *
   * - To get (or set) the current time step's recurrent state, use
   *   `RecurrentState(CurrentStep())`.
   * - To get the previous time step's recurrent state, use
   *   `RecurrentState(PreviousStep())`.
   */
  const MatType& RecurrentState(const size_t t) const;
  // Modify the stored recurrent state at time step `t`.  Be careful!
  MatType& RecurrentState(const size_t t);

  /**
   * Get the stored recurrent gradient at the given time step `t`.  `t` must be
   * `CurrentStep()` or `PreviousStep()`.  The recurrent gradient represents the
   * gradient of the output of the network with respect to the hidden state.
   */
  const MatType& RecurrentGradient(const size_t t) const;
  // Modify the stored recurrent gradient at time step `t`.  Be careful!
  MatType& RecurrentGradient(const size_t t);

  // Get the current step index to use in a forward or backward pass.
  size_t CurrentStep() const { return currentStep; }
  // Modify the current step index to use in a forward or backward pass.
  // (Don't do this inside of your recurrent layer's implementation!  This is
  // meant to be done by the enclosing network.)
  void CurrentStep(const size_t& step, const bool end = false);

  // Get the previous step.  This is a very simple function but can lead to
  // slightly more readable code in Forward(), Backward(), and Gradient()
  // implementations.
  size_t PreviousStep() const { return currentStep - 1; }

  // Get whether or not recurrent state has been computed for previous time
  // steps.  If not, no previous recurrent state should be used in the
  // computation.
  bool HasPreviousStep() const { return currentStep != size_t(0); }

  // Get whether or not the current time step is the final time step.  If so,
  // then Backward() and Gradient() should not use RecurrentGradient().
  bool AtFinalStep() const { return atFinalStep; }

  //! Serialize the recurrent layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // The current time step index.  This is set by the enclosing network during
  // forward and backward passes.
  size_t currentStep;

  // If true, then there are no further time steps.
  bool atFinalStep;

  // This holds the recurrent state at each time step for BPTT.  If BPTT is not
  // being used (e.g. if we are only running the network in forward mode and not
  // training), then only one previous time step is held.
  arma::Cube<typename MatType::elem_type> recurrentState;
  // This holds the recurrent gradient for BPTT.  If BPTT is not being used,
  // this is empty.
  arma::Cube<typename MatType::elem_type> recurrentGradient;
};

} // namespace mlpack

#include "recurrent_layer_impl.hpp"

#endif
