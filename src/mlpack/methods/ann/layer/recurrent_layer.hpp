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

namespace mlpack {
namespace ann {

template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class RecurrentLayer : public Layer<InputType, OutputType>
{
 public:
  RecurrentLayer();

  virtual ~RecurrentLayer() { }

  RecurrentLayer(const RecurrentLayer& other);
  RecurrentLayer(RecurrentLayer&& other);
  RecurrentLayer& operator=(const RecurrentLayer& other);
  RecurrentLayer& operator=(RecurrentLayer&& other);

  // TODO: document, but downstream must overload this
  virtual void ClearRecurrentState(
      const size_t bpttSteps,
      const size_t batchSize) = 0;

  size_t CurrentStep() const { return currentStep; }
  size_t& CurrentStep() { return currentStep; }
  size_t PreviousStep() const { return previousStep; }
  size_t& PreviousStep() { return previousStep; }

  bool HasPreviousStep() const { return previousStep != size_t(-1); }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  size_t currentStep;
  size_t previousStep;
};

} // namespace ann
} // namespace mlpack

#include "recurrent_layer_impl.hpp"

#endif
