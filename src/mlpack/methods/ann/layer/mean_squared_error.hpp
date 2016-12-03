/**
 * @file mean_squared_error.hpp
 * @author Marcus Edel
 *
<<<<<<< HEAD
 * Definition of the mean squared error performance function.
=======
 * Definition and implementation of the mean squared error performance function.
>>>>>>> Refactor ann layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_SQUARED_ERROR_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_SQUARED_ERROR_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The mean squared error performance function measures the network's
 * performance according to the mean of squared errors.
 *
 * @tparam ActivationFunction Activation function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class MeanSquaredError
{
 public:
  /**
   * Create the MeanSquaredError object.
   */
<<<<<<< HEAD
  MeanSquaredError();
=======
  MeanSquaredError() { /* Nothing to do here. */ }
>>>>>>> Refactor ann layer.

  /*
   * Computes the mean squared error function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
<<<<<<< HEAD
  double Forward(const arma::Mat<eT>&& input, const arma::Mat<eT>&& target);
=======
  double Forward(const arma::Mat<eT>&& input, const arma::Mat<eT>&& target)
  {
    return arma::mean(arma::mean(arma::square(input - target)));
  }

>>>>>>> Refactor ann layer.
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>&& input,
                const arma::Mat<eT>&& target,
<<<<<<< HEAD
                arma::Mat<eT>&& output);
=======
                arma::Mat<eT>&& output)
  {
    output = (input - target);
  }
>>>>>>> Refactor ann layer.

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

<<<<<<< HEAD
  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

=======
>>>>>>> Refactor ann layer.
 private:
  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MeanSquaredError

<<<<<<< HEAD
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "mean_squared_error_impl.hpp"
=======
}; // namespace ann
}; // namespace mlpack
>>>>>>> Refactor ann layer.

#endif
