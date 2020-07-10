/**
 * @file methods/ann/layer/subview.hpp
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
   * @param inSize Width of sample.
   * @param beginRow Starting row index.
   * @param endRow Ending row index.
   * @param beginCol Starting column index.
   * @param endCol Ending column index.
   */
  Subview(const size_t inSize = 1,
          const size_t beginRow = 0,
          const size_t endRow = 0,
          const size_t beginCol = 0,
          const size_t endCol = 0) :
      inSize(inSize),
      beginRow(beginRow),
      endRow(endRow),
      beginCol(beginCol),
      endCol(endCol)
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
  void Forward(const InputType& input, OutputType& output)
  {
    size_t batchSize = input.n_cols / inSize;

    // Check if subview parameters are within the indices of input sample.
    endRow = ((endRow < input.n_rows) && (endRow >= beginRow))?
        endRow : (input.n_rows - 1);
    endCol = ((endCol < inSize) && (endCol >= beginCol)) ?
        endCol : (inSize - 1);

    output.set_size(
        (endRow - beginRow + 1) * (endCol - beginCol + 1), batchSize);

    size_t batchBegin = beginCol;
    size_t batchEnd = endCol;

    // Check whether the input is already in desired form.
    if ((input.n_rows != ((endRow - beginRow + 1) *
        (endCol - beginCol + 1))) || (input.n_cols != batchSize))
    {
      for (size_t i = 0; i < batchSize; ++i)
      {
        output.col(i) = arma::vectorise(
            input.submat(beginRow, batchBegin, endRow, batchEnd));

        // Move to next batch.
        batchBegin += inSize;
        batchEnd += inSize;
      }
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
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    g = gy;
  }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the width of each sample.
  size_t InSize() const { return inSize; }

  //! Get the starting row index of subview vector or matrix.
  size_t const& BeginRow() const { return beginRow; }
  //! Modify the width of each sample.
  size_t& BeginRow() { return beginRow; }

  //! Get the ending row index of subview vector or matrix.
  size_t const& EndRow() const { return endRow; }
  //! Modify the width of each sample.
  size_t& EndRow() { return endRow; }

  //! Get the width of each sample.
  size_t const& BeginCol() const { return beginCol; }
  //! Modify the width of each sample.
  size_t& BeginCol() { return beginCol; }

  //! Get the ending column index of subview vector or matrix.
  size_t const& EndCol() const { return endCol; }
  //! Modify the width of each sample.
  size_t& EndCol() { return endCol; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(inSize);
    ar & BOOST_SERIALIZATION_NVP(beginRow);
    ar & BOOST_SERIALIZATION_NVP(endRow);
    ar & BOOST_SERIALIZATION_NVP(beginCol);
    ar & BOOST_SERIALIZATION_NVP(endCol);
  }

 private:
  //! Width of each sample.
  size_t inSize;

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

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class Subview

} // namespace ann
} // namespace mlpack

#endif
