/**
 * @file select.hpp
 * @author Marcus Edel
 *
<<<<<<< HEAD
<<<<<<< HEAD
 * Definition of the Select module.
=======
 * Definition and implementation of the Select module.
>>>>>>> Refactor ann layer.
=======
 * Definition of the Select module.
>>>>>>> Split layer modules into definition and implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SELECT_HPP
#define MLPACK_METHODS_ANN_LAYER_SELECT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The select module selects the specified column from a given input matrix.
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
class Select
{
 public:
  /**
   * Create the Select object.
   *
   * @param index The column which should be extracted from the given input.
<<<<<<< HEAD
<<<<<<< HEAD
   * @param elements The number of elements that should be used.
   */
  Select(const size_t index, const size_t elements = 0);
<<<<<<< HEAD
=======
   * @param index The number of elements that should be used.
=======
   * @param elements The number of elements that should be used.
>>>>>>> Minor style fixes.
   */
  Select(const size_t index, const size_t elements = 0) :
      index(index),
      elements(elements)
  {
    /* Nothing to do here. */
  }
>>>>>>> Refactor ann layer.
=======
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
    if (elements == 0)
    {
      output = input.col(index);
    }
    else
    {
      output = input.submat(0, index, elements - 1, index);
    }
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
                arma::Mat<eT>&& gy,
<<<<<<< HEAD
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    if (elements == 0)
    {
      g = gy;
    }
    else
    {
      g = gy.submat(0, 0, elements - 1, 0);
    }
  }
>>>>>>> Refactor ann layer.
=======
                arma::Mat<eT>&& g);
>>>>>>> Split layer modules into definition and implementation.

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
<<<<<<< HEAD
=======
>>>>>>> Split layer modules into definition and implementation.
  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

<<<<<<< HEAD
=======
>>>>>>> Refactor ann layer.
=======
>>>>>>> Split layer modules into definition and implementation.
 private:
  //! Locally-stored column index.
  size_t index;

  //! Locally-stored number of elements selected.
  size_t elements;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Select

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> Split layer modules into definition and implementation.
} // namespace ann
} // namespace mlpack

// Include implementation.
#include "select_impl.hpp"
<<<<<<< HEAD
=======
}; // namespace ann
}; // namespace mlpack
>>>>>>> Refactor ann layer.
=======
>>>>>>> Split layer modules into definition and implementation.

#endif
