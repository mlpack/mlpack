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
 * `ClearRecurrentState(bpttSteps, batchSize)` function; this function should
 * allocate space to store previous states with the given batch size.  See the
 * documentation for that function for more details.
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

  //! Copy the given RecurrentLayer.
  RecurrentLayer(const RecurrentLayer& other);
  //! Take ownership of the given RecurrentLayer.
  RecurrentLayer(RecurrentLayer&& other);
  //! Copy the given RecurrentLayer.
  RecurrentLayer& operator=(const RecurrentLayer& other);
  //! Take ownership of the given RecurrentLayer.
  RecurrentLayer& operator=(RecurrentLayer&& other);

  /**
   * ClearRecurrentState() is called before any forward pass of a recurrent
   * network.  This function is responsible for allocating any memory necessary
   * to store `bpttSteps` steps of previous forward and backward passes, with a
   * batch size of `batchSize`.
   *
   * Any internal state of the recurrent layer should be set to 0.
   */
  virtual void ClearRecurrentState(
      const size_t bpttSteps,
      const size_t batchSize) = 0;

  //! Get the current step index to use in a forward or backward pass.
  size_t CurrentStep() const { return currentStep; }
  //! Modify the current step index to use in a forward or backward pass.
  //! (Don't do this inside of your recurrent layer's implementation!  This is
  //! meant to be done by the enclosing network.)
  size_t& CurrentStep() { return currentStep; }

  //! Get the previous step index, representing the value of CurrentStep() in
  //! the previous call to Forward() or Backward().
  size_t PreviousStep() const { return previousStep; }
  //! Modify the previous step index, representing the value of CurrentStep() in
  //! the previous call to Forward() or Backward().  (Don't modify this inside
  //! of your recurrent layer's implementation!  This is meant to be done by the
  //! enclosing network.)
  size_t& PreviousStep() { return previousStep; }

  //! If Forward() or Backward() has been called since ClearRecurrentState(),
  //! this will return true.  This should be used to determine if recurrent
  //! state should be considered in computations.
  bool HasPreviousStep() const { return previousStep != size_t(-1); }

  //! Serialize the recurrent layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The current index of the step.  This is set by the enclosing network
  //! during forward and backward passes.
  size_t currentStep;
  //! The previous index of the step.  This is set by the enclosing network
  //! during forward and backward passes.
  size_t previousStep;
};

} // namespace mlpack

#include "recurrent_layer_impl.hpp"

#endif
