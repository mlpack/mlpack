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

#include "layer.hpp"

namespace mlpack {

/**
 * Implementation of the subview layer. The subview layer modifies the input to
 * a submatrix of required size.
 *
 * @tparam InputType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
class SubviewType : public Layer<InputType, OutputType>
{
 public:
  /**
   * Create the Subview layer object using the specified range of input to
   * accept.
   *
   * @param beginRow Starting row index.
   * @param endRow Ending row index.
   * @param beginCol Starting column index.
   * @param endCol Ending column index.
   */
  SubviewType(const size_t beginRow = 0,
              const size_t endRow = 0,
              const size_t beginCol = 0,
              const size_t endCol = 0) :
      beginRow(beginRow),
      endRow(endRow),
      beginCol(beginCol),
      endCol(endCol)
  {
    /* Nothing to do here */
  }

  //! Clone the SubviewType object. This handles polymorphism correctly.
  SubviewType* Clone() const { return new SubviewType(*this); }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const InputType& input, OutputType& output)
  {
    size_t batchSize = input.n_cols;

    // Check if subview parameters are within the indices of input sample.
    // TODO: this seems incorrect
    endRow = ((endRow < inputDimensions[0]) && (endRow >= beginRow))?
        endRow : (inputDimensions[0] - 1);
    endCol = ((endCol < inputDimensions[1]) && (endCol >= beginCol)) ?
        endCol : (inputDimensions[1] - 1);

    // TODO: this is maybe not right?
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
        output.col(i) = vectorise(
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
  void Backward(const InputType& /* input */,
                const OutputType& gy,
                OutputType& g)
  {
    g = gy;
  }

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

  const std::vector<size_t> OutputDimensions() const
  {
    // TODO: relax this restriction
    for (size_t i = 2; i < inputDimensions.size(); ++i)
    {
      if (inputDimensions[i] > 1)
      {
        throw std::invalid_argument("Subview(): layer input must be two-"
            "dimensional!");
      }
    }

    std::vector<size_t> outputDimensions(inputDimensions);
    outputDimensions[0] = (endRow - beginRow + 1);
    outputDimensions[1] = (endCol - beginCol + 1);

    return outputDimensions;
  }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(cereal::base_class<Layer<InputType, OutputType>>(this));

    ar(CEREAL_NVP(beginRow));
    ar(CEREAL_NVP(endRow));
    ar(CEREAL_NVP(beginCol));
    ar(CEREAL_NVP(endCol));
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
}; // class SubviewType

// Standard Subview layer.
using Subview = SubviewType<arma::mat, arma::mat>;

} // namespace mlpack

#endif
