/**
 * @file subview.hpp
 * @author Haritha Nair
 *
 * Definition of the Subview class, which modifies the input as necessary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_SUBVIEW_HPP
#define MLPACK_METHODS_ANN_LAYER_SUBVIEW_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>

namespace mlpack {
namespace ann {

/**
 * Implementation of the subview layer. The subview layer modifies the input to
 * a submatrix of required size.
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
class Subview
{
 public:
  /**
   * Create the Subview layer object using the specified range of input to
   * accept.
   *
   * @param begin Start index.
   * @param end End index.
   */
  Subview(const size_t beginRow = 0,
          const size_t endRow = 0,
          const size_t beginCol = 0,
          const size_t endCol = 0) :
      beginRow(beginRow), endRow(endRow), beginCol(beginCol), endCol(endCol)
  {
    /* Nothing to do here */
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename InputType, typename OutputType>
  void Forward(InputType&& input, OutputType&& output)
  {
    // Check if input has been selected as required.
    if (((input.n_rows != (endRow - beginRow + 1)) ||
        (input.n_cols != (endCol - beginCol +1)))
        && !(endRow == 0 && endCol == 0))
    {
      output = input.submat(beginRow, beginCol, endRow, endCol);
    }
    else
    {
      output = input;
    }
  }

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
  void Backward(arma::Mat<eT>&& /* input */,
                arma::Mat<eT>&& gy,
                arma::Mat<eT>&& g)
  {
    g = gy;
  }

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
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(beginRow);
    ar & BOOST_SERIALIZATION_NVP(endRow);
    ar & BOOST_SERIALIZATION_NVP(beginCol);
    ar & BOOST_SERIALIZATION_NVP(endCol);
  }

 private:
  //! Starting row index of subview vector or matrix.
  size_t beginRow;

  //! Ending row index of subview vector or matrix.
  size_t endRow;

  //! Starting column index of subview vector or matrix.
  size_t beginCol;

  //! Ending column index of subview vector or matrix.
  size_t endCol;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Subview

} // namespace ann
} // namespace mlpack

#endif
