/**
 * @file linear.hpp
 * @author Marcus Edel
 *
 * Definition of the LinearNoBias class also known as fully-connected layer or
 * affine transformation without the bias term.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_HPP
#define MLPACK_METHODS_ANN_LAYER_LINEAR_NO_BIAS_HPP

#include <mlpack/core.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the LinearNoBias class. The LinearNoBias class represents a
 * single layer of a neural network.
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
class LinearNoBias
{
 public:
  //! Create the LinearNoBias object.
<<<<<<< HEAD
  LinearNoBias();
=======
  LinearNoBias() {}
>>>>>>> Refactor ann layer.
  /**
   * Create the LinearNoBias object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
<<<<<<< HEAD
  LinearNoBias(const size_t inSize, const size_t outSize);
=======
  LinearNoBias(const size_t inSize, const size_t outSize) :
      inSize(inSize),
      outSize(outSize)
  {
    weights.set_size(outSize * inSize, 1);
  }
>>>>>>> Refactor ann layer.

  /*
   * Reset the layer parameter.
   */
<<<<<<< HEAD
  void Reset();
=======
  void Reset()
  {
    weight = arma::mat(weights.memptr(), outSize, inSize, false, false);
  }
>>>>>>> Refactor ann layer.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);
=======
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output)
  {
    output = weight * input;
  }
>>>>>>> Refactor ann layer.

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
                arma::Mat<eT>&& gy,
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    g = weight.t() * gy;
  }
>>>>>>> Refactor ann layer.

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
                arma::Mat<eT>&& gradient);
=======
                arma::Mat<eT>&& gradient)
  {
    gradient.submat(0, 0, weight.n_elem - 1, 0) = arma::vectorise(
        error * input.t());
  }
>>>>>>> Refactor ann layer.

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
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(weights, "weights");
    ar & data::CreateNVP(inSize, "inSize");
    ar & data::CreateNVP(outSize, "outSize");
  }
>>>>>>> Refactor ann layer.

 private:

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored weight parameter.
  OutputDataType weight;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class LinearNoBias

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
// Include implementation.
#include "linear_no_bias_impl.hpp"

=======
>>>>>>> Refactor ann layer.
#endif
