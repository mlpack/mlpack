/**
 * @file methods/ann/activation_functions/elish_function.hpp
 * @author Bisakh Mondal
 *
 * Definition and implementation of the ELiSH function as described by
 * Mina Basirat and Peter M. Roth.
 *
 * For more information see the following paper
 *
 * @code
 * @misc{Basirat2018,
 *    title = {The Quest for the Golden Activation Function},
 *    author = {Mina Basirat and Peter M. Roth},
 *    year = {2018},
 *    url = {https://arxiv.org/pdf/1808.00783.pdf},
 *    eprint = {1808.00783},
 *    archivePrefix = {arXiv},
 *    primaryClass = {cs.NE} }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ELISH_FUNCTION_HPP
#define MLPACK_METHODS_ANN_LAYER_ELISH_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

#include "layer.hpp"

namespace mlpack {

/**
 * The ELiSH function, defined by
 *
 * @f{eqnarray*}{
 *   f(x) &=& \begin{cases}
 *      x / (1 + e^{-x}) & x \geq 0\\
 *     (e^{x} - 1) / (1 + e^{-x}) & x < 0.\\
 *   \end{cases} \\
 *   f'(x) &=& \begin{cases}
 *      1 / (1 + e^{-x}) + x * e^{-x} / (1 + e^{-x})^2 & x \geq 0\\
 *      e^x - 2 / (1 + e^x) + 2 / (1 + e^x)^2 & x < 0.\\
 *   \end{cases}
 * @f}
 */
template<typename MatType = arma::mat>
class ElishType : public Layer<MatType>
{
 public:
  /**
   * Create the ElishType object.
   */
  ElishType() { }

  //! Clone the ElishType object. This handles polymorphism correctly.
  ElishType* Clone() const { return new ElishType(*this); }

  // Virtual destructor.
  virtual ~ElishType() { }

  //! Copy the given ElishType
  ElishType(const ElishType& other);
  //! Take ownership of the given ElishType.
  ElishType(ElishType&& other);
  //! Copy the given ElishType.
  ElishType& operator=(const ElishType& other);
  //! Take ownership of the given ElishType.
  ElishType& operator=(ElishType&& other);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& input, const MatType& gy, MatType& g);

  //! Serialize the layer.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
  
 private:
  //! Locally stored first derivative of the activation function.
  MatType derivative;

}; // class ElishType

// Convenience typedefs.

// Standard ElishType layer.
typedef ElishType<arma::mat> Elish;

} // namespace mlpack

// Include implementation.
#include "elish_function_impl.hpp"

#endif