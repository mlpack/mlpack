/**
 * @file lookup.hpp
 * @author Marcus Edel
 *
 * Definition of the Lookup class a particular convolution, where the width of
 * the convolution is 1.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP
#define MLPACK_METHODS_ANN_LAYER_LOOKUP_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Lookup class. The Lookup class is a particular
 * convolution, where the width of the convolution is 1.
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
class Lookup
{
 public:
  /**
   * Create the Lookup object using the specified number of input and output
   * units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
<<<<<<< HEAD
<<<<<<< HEAD
  Lookup(const size_t inSize, const size_t outSize);
=======
  Lookup(const size_t inSize, const size_t outSize) :
      inSize(inSize),
      outSize(outSize)
  {
    weights.set_size(outSize, inSize);
  }
>>>>>>> Refactor ann layer.
=======
  Lookup(const size_t inSize, const size_t outSize);
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
<<<<<<< HEAD
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    output = weights.cols(arma::conv_to<arma::uvec>::from(input) - 1);
  }
>>>>>>> Refactor ann layer.
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& /* input */,
                const arma::Mat<eT>&& gy,
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    g = gy;
  }
>>>>>>> Refactor ann layer.
=======
                arma::Mat<eT>&& g);
>>>>>>> Split layer modules into definition and implementation.

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>&& input,
                arma::Mat<eT>&& error,
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& gradient);
=======
                arma::Mat<eT>&& gradient)
  {
    gradient = arma::zeros<arma::Mat<eT> >(weights.n_rows, weights.n_cols);
    gradient.cols(arma::conv_to<arma::uvec>::from(input) - 1) = error;
  }
>>>>>>> Refactor ann layer.
=======
                arma::Mat<eT>&& gradient);
>>>>>>> Split layer modules into definition and implementation.

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
<<<<<<< HEAD
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(inSize, "inSize");
    ar & data::CreateNVP(outSize, "outSize");
  }
>>>>>>> Refactor ann layer.
=======
  void Serialize(Archive& ar, const unsigned int /* version */);
>>>>>>> Split layer modules into definition and implementation.

 private:

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Lookup

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
<<<<<<< HEAD
// Include implementation.
#include "lookup_impl.hpp"

=======
>>>>>>> Refactor ann layer.
=======
// Include implementation.
#include "lookup_impl.hpp"

>>>>>>> Split layer modules into definition and implementation.
#endif
