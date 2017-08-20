/**
 * @file resize.hpp
 * @author Kris Singh
 *
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_RESIZE_HPP
#define MLPACK_METHODS_ANN_LAYER_RESIZE_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/image_functions/bilinear_function.hpp>

#include "layer_types.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Resize layer class. The Resize class represents a
 * single layer of a neural network.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 *
 * @tparam InterpolationType Type of the inpterpolation applied to the input
 *          (BilinearFunction)
 *
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat,
    class InterpolationType = BiLinearFunction
>
class Resize
{
 public:
  //! Create the Resize object.
  Resize(InterpolationType policy);

  /**
   * Forward pass through the Resize layer. The resize layer interpolates 
   * the matrix using the given interpolation method.
   * If the size of the input and output are same the Forward layer
   * does no nothing.
   *
   * @param input Input the input matrix to interpolate
   * @param output The interpolated matrix.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>&& input, arma::Mat<eT>&& output);

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
                arma::Mat<eT>&& g);

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

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Locally-stored interpolation method
  InterpolationType policy;
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

}; // class Resize

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "resize_impl.hpp"

#endif
