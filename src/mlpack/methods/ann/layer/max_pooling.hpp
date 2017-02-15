/**
 * @file max_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MaxPooling class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MAX_POOLING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/*
 * The max pooling rule for convolution neural networks. Take the maximum value
 * within the receptive block.
 */
class MaxPoolingRule
{
 public:
  /*
   * Return the maximum value within the receptive block.
   *
   * @param input Input used to perform the pooling operation.
   */
  template<typename MatType>
  size_t Pooling(const MatType& input)
  {
    return arma::as_scalar(arma::find(input.max() == input, 1));
  }
};

/**
 * Implementation of the MaxPooling layer.
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
class MaxPooling
{
public:
  //! Create the MaxPooling object.
  MaxPooling();

  /**
   * Create the MaxPooling object using the specified number of units.
   *
   * @param kW Width of the pooling window.
   * @param kH Height of the pooling window.
   * @param dW Width of the stride operation.
   * @param dH Width of the stride operation.
   * @param floor Rounding operator (floor or ceil).
   */
  MaxPooling(const size_t kW,
             const size_t kH,
             const size_t dW = 1,
             const size_t dH = 1,
             const bool floor = true);

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
   * Ordinary feed backward pass of a neural network, using 3rd-order tensors as
   * input, calculating the function f(x) by propagating x backwards through f.
   * Using the results from the feed forward pass.
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

  //! Get the width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the height.
  size_t const& InputHeight() const { return inputHeight; }
  //! Modify the height.
  size_t& InputHeight() { return inputHeight; }

  //! Get the width.
  size_t const& OutputWidth() const { return outputWidth; }
  //! Modify the width.
  size_t& OutputWidth() { return outputWidth; }

  //! Get the height.
  size_t const& OutputHeight() const { return outputHeight; }
  //! Modify the height.
  size_t& OutputHeight() { return outputHeight; }

  //! Get the value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:

 /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices.
   */
  template<typename eT>
  void PoolingOperation(const arma::Mat<eT>& input,
                        arma::Mat<eT>& output,
                        arma::Mat<eT>& poolingIndices)
  {
    for (size_t j = 0, colidx = 0; j < output.n_cols; ++j, colidx += dW)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows; ++i, rowidx += dH)
      {
        arma::mat subInput = input(arma::span(rowidx, rowidx + kW - 1 - offset),
            arma::span(colidx, colidx + kH - 1 - offset));

        const size_t idx = pooling.Pooling(subInput);
        output(i, j) = subInput(idx);

        if (!deterministic)
        {
          arma::Mat<size_t> subIndices = indices(arma::span(rowidx,
              rowidx + kW - 1 - offset),
              arma::span(colidx, colidx + kH - 1 - offset));

          poolingIndices(i, j) = subIndices(idx);
        }
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param error The backward error.
   * @param output The pooled result.
   * @param poolingIndices The pooled indices.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& error,
                 arma::Mat<eT>& output,
                 arma::Mat<eT>& poolingIndices)
  {
    for (size_t i = 0; i < poolingIndices.n_elem; ++i)
    {
      output(poolingIndices(i)) += error(i);
    }
  }

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored width of the pooling window.
  size_t kW;

  //! Locally-stored height of the pooling window.
  size_t kH;

  //! Locally-stored width of the stride operation.
  size_t dW;

  //! Locally-stored height of the stride operation.
  size_t dH;

  //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! Rounding operation used.
  bool floor;

  //! Locally-stored stored rounding offset.
  size_t offset;

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

  //! If true use maximum a posteriori during the forward pass.
  bool deterministic;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored pooling strategy.
  MaxPoolingRule pooling;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored indices matrix parameter.
  arma::Mat<size_t> indices;

  //! Locally-stored indices column parameter.
  arma::Col<size_t> indicesCol;

  //! Locally-stored pooling indicies.
  std::vector<arma::cube> poolingIndices;
}; // class MaxPooling

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "max_pooling_impl.hpp"

#endif
