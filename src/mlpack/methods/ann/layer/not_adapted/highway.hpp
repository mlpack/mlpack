// Temporarily drop.
/**
 * @file methods/ann/layer/highway.hpp
 * @author Konstantin Sidorov
 * @author Saksham Bansal
 *
 * Definition of the Highway layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP
#define MLPACK_METHODS_ANN_LAYER_HIGHWAY_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the Highway layer.  The Highway class can vary its behavior
 * between that of feed-forward fully connected network container and that
 * of a layer which simply passes its inputs through depending on the transform
 * gate. Note that the size of the input and output matrices of this class
 * should be equal.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{Srivastava2015,
 *   author  = {Rupesh Kumar Srivastava, Klaus Greff, Jurgen Schmidhuber},
 *   title   = {Training Very Deep Networks},
 *   journal = {Advances in Neural Information Processing Systems},
 *   year    = {2015},
 *   url     = {https://arxiv.org/abs/1507.06228},
 * }
 * @endcode
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class HighwayType : public MultiLayer<InputType, OutputType>
{
 public:
  //! Create the HighwayType object.
  HighwayType();

  //! Destroy the HighwayType object.
  virtual ~HighwayType();

  //! Clone the HighwayType object. This handles polymorphism correctly.
  HighwayType* Clone() const { return new HighwayType(*this); }

  //! Copy the given HighwayType (but not weights).
  HighwayType(const HighwayType& other);
  //! Take ownership of the given HighwayType (but not weights).
  HighwayType(HighwayType&& other);
  //! Copy the given HighwayType (but not weights).
  HighwayType& operator=(const HighwayType& other);
  //! Take ownership of the given HighwayType (but not weights).
  HighwayType& operator=(HighwayType&& other);

  void SetWeights(typename OutputType::elem_type* weightsPtr);

  /**
   * Ordinary feed-forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output);

  /**
   * Ordinary feed-backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the
   * feed-forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const InputType& input,
                const OutputType& error,
                OutputType& gradient);

  //! Get the parameters.
  OutputType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputType& Parameters() { return weights; }

  //! Get the number of trainable weights.
  size_t WeightSize() const
  {
    size_t result = this->totalInputSize * (this->totalInputSize + 1);
    for (size_t i = 0; i < this->network.size(); ++i)
      result += this->network[i]->WeightSize();
    return result;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Locally-stored weight object.
  OutputType weights;

  //! Weights for transformation of output.
  OutputType transformWeight;

  //! Bias for transformation of output.
  OutputType transformBias;

  //! Locally-stored transform gate parameters.
  OutputType transformGate;

  //! Locally-stored transform gate activation.
  OutputType transformGateActivation;

  //! Locally-stored transform gate error.
  OutputType transformGateError;
}; // class HighwayType

// Standard Highway layer.
using Highway = HighwayType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "highway_impl.hpp"

#endif
