/**
 * @file join.hpp
 * @author Marcus Edel
 *
 * Definition of the Join module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_JOIN_HPP
#define MLPACK_METHODS_ANN_LAYER_JOIN_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the Join module class. The Join class accumulates
 * the output of various modules.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template<
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class Join
{
 public:
  //! Create the Join object.
<<<<<<< HEAD
<<<<<<< HEAD
  Join();
=======
  Join()
  {
    // Nothing to do here.
  }
>>>>>>> Refactor ann layer.
=======
  Join();
>>>>>>> Split layer modules into definition and implementation.

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
<<<<<<< HEAD
<<<<<<< HEAD
  void Forward(const InputType&& input, OutputType&& output);
=======
  void Forward(const InputType&& input, OutputType&& output)
  {
    inSizeRows = input.n_rows;
    inSizeCols = input.n_cols;
    output = arma::vectorise(input);
  }
>>>>>>> Refactor ann layer.
=======
  void Forward(const InputType&& input, OutputType&& output);
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
                arma::Mat<eT>&& gy,
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    g = arma::mat(gy.memptr(), inSizeRows, inSizeCols, false, false);
  }
>>>>>>> Refactor ann layer.
=======
                arma::Mat<eT>&& g);
>>>>>>> Split layer modules into definition and implementation.

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
   * Serialize the layer.
   */
  template<typename Archive>
<<<<<<< HEAD
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(inSizeRows, "inSizeRows");
    ar & data::CreateNVP(inSizeCols, "inSizeCols");
  }
>>>>>>> Refactor ann layer.
=======
  void Serialize(Archive& ar, const unsigned int /* version */);
>>>>>>> Split layer modules into definition and implementation.

 private:
  //! Locally-stored number of input rows.
  size_t inSizeRows;

  //! Locally-stored number of input cols.
  size_t inSizeCols;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Join

} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
<<<<<<< HEAD
// Include implementation.
#include "join_impl.hpp"

=======
>>>>>>> Refactor ann layer.
=======
// Include implementation.
#include "join_impl.hpp"

>>>>>>> Split layer modules into definition and implementation.
#endif
