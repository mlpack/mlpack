/**
 * @file methods/ann/layer/multiply_merge.hpp
 * @author Haritha Nair
 *
 * Definition of the MultiplyMerge module which multiplies the output of the
 * given modules element-wise.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP
#define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_HPP

#include <mlpack/prereqs.hpp>

// #include "../visitor/delete_visitor.hpp"
// #include "../visitor/delta_visitor.hpp"
// #include "../visitor/output_parameter_visitor.hpp"

#include "layer_types.hpp"

namespace mlpack {

/**
 * Implementation of the MultiplyMerge module class. The MultiplyMerge class
 * multiplies the output of various modules element-wise.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class MultiplyMergeType : public MultiLayer<InputType, OutputType>
{
 public:
  /**
   * Create the MultiplyMerge object using the specified parameters.
   *
   * @param model Expose all the network modules.
   * @param run Call the Forward/Backward method before the output is merged.
   */
  MultiplyMergeType(const bool model = false, const bool run = true);

  //! Copy Constructor.
  MultiplyMerge(const MultiplyMerge& layer);

  //! Move Constructor.
  MultiplyMerge(MultiplyMerge&& layer);

  //! Copy assignment operator.
  MultiplyMerge& operator=(const MultiplyMerge& layer);

  //! Move assignment operator.
  MultiplyMerge& operator=(MultiplyMerge&& layer);

  //! Destructor to release allocated memory.
  ~MultiplyMergeType();

  //! Clone the MultiplyMergeType object. This handles polymorphism correctly.
  MultiplyMergeType* Clone() const { return new MultiplyMergeType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param * (input) Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& /* input */, OutputType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f, using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g);

  /*
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

  //! Get the size of the weights.
  size_t WeightSize() const { return 0; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Parameter which indicates if the Forward/Backward method should be called
  //! before merging the output.
  bool run;

  //! We need this to know whether we should delete the layer in the destructor.
  bool ownsLayer;

  //! Locally-stored weight object.
  OutputType weights;
}; // class MultiplyMergeType

// Standard MultiplyMerge layer.
using MultiplyMerge = MultiplyMergeType<arma::mat, arma::mat>;

} // namespace mlpack

// Include implementation.
#include "multiply_merge_impl.hpp"

#endif
