/**
 * @file concatenate.hpp
 * @author Atharva Khandait
 *
 * Definition of the Concatenate class that concatenate a constant matrix to
 * the incoming data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCATENATE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Concatenate module class. The Concatenate module
 * concatenates a constant given matrix to the incoming data.
 * Note: Users need to use the Concat() function to provide the concat matrix.
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
class Concatenate
{
 public:
  /**
   * Create the Concatenate object using the specified number of output units.
   */
  Concatenate();

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
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
                const arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g);

  //! Get the parameters.
  OutputDataType const& Parameters() const { return weights; }
  //! Modify the parameters.
  OutputDataType& Parameters() { return weights; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the concat matrix.
  OutputDataType const& Concat() const { return concat; }
  //! Modify the delta.
  OutputDataType& Concat() { return concat; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    // Nothing to do here.
  }

 private:
  //! Locally-stored number of input rows.
  size_t inRows;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored matrix to be concatenated to input.
  OutputDataType concat;
}; // class Concatenate

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concatenate_impl.hpp"

#endif
