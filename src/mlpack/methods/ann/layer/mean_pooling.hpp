/**
 * @file mean_pooling.hpp
 * @author Marcus Edel
 * @author Nilay Jain
 *
 * Definition of the MeanPooling layer class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP
#define MLPACK_METHODS_ANN_LAYER_MEAN_POOLING_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the MeanPooling.
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
class MeanPooling
{
public:
  //! Create the MeanPooling object.
<<<<<<< HEAD
  MeanPooling();
=======
  MeanPooling()
  {
    /* Nothing to do here */
  }
>>>>>>> Refactor ann layer.

  /**
   * Create the MeanPooling object using the specified number of units.
   *
   * @param kW Width of the pooling window.
   * @param kH Height of the pooling window.
   * @param dW Width of the stride operation.
   * @param dH Width of the stride operation.
   */
  MeanPooling(const size_t kW,
<<<<<<< HEAD
              const size_t kH,
              const size_t dW = 1,
              const size_t dH = 1,
              const bool floor = true);
=======
          const size_t kH,
          const size_t dW = 1,
          const size_t dH = 1,
          const bool floor = true) :
      kW(kW),
      kH(kH),
      dW(dW),
      dH(dH),
      inputWidth(0),
      inputHeight(0),
      outputWidth(0),
      outputHeight(0),
      reset(false),
      floor(floor),
      deterministic(false),
      offset(0)

  {
    /* Nothing to do here. */
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
    size_t slices = input.n_elem / (inputWidth * inputHeight);
    inputTemp = arma::cube(input.memptr(), inputWidth, inputHeight, slices);

    if (floor)
    {
      outputWidth = std::floor((inputWidth - (double) kW) / (double) dW + 1);
      outputHeight = std::floor((inputHeight - (double) kH) / (double) dH + 1);

      offset = 0;
    }
    else
    {
      outputWidth = std::ceil((inputWidth - (double) kW) / (double) dW + 1);
      outputHeight = std::ceil((inputHeight - (double) kH) / (double) dH + 1);

      offset = 1;
    }

    outputTemp = arma::zeros<arma::Cube<eT> >(outputWidth, outputHeight,
        slices);

    for (size_t s = 0; s < inputTemp.n_slices; s++)
    {

      Pooling(inputTemp.slice(s), outputTemp.slice(s));
    }

    output = arma::Mat<eT>(outputTemp.memptr(), outputTemp.n_elem, 1);

    outputWidth = outputTemp.n_rows;
    outputHeight = outputTemp.n_cols;
    outSize = slices;
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
                arma::Mat<eT>&& g);
=======
                arma::Mat<eT>&& g)
  {
    arma::cube mappedError = arma::cube(gy.memptr(), outputWidth,
        outputHeight, outSize);

    gTemp = arma::zeros<arma::cube>(inputTemp.n_rows,
        inputTemp.n_cols, inputTemp.n_slices);

    for (size_t s = 0; s < mappedError.n_slices; s++)
    {
      Unpooling(inputTemp.slice(s), mappedError.slice(s), gTemp.slice(s));
    }

    g = arma::mat(gTemp.memptr(), gTemp.n_elem, 1);
  }
>>>>>>> Refactor ann layer.

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
<<<<<<< HEAD
  void Serialize(Archive& ar, const unsigned int /* version */);
=======
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(kW, "kW");
    ar & data::CreateNVP(kH, "kH");
    ar & data::CreateNVP(dW, "dW");
    ar & data::CreateNVP(dH, "dH");
  }
>>>>>>> Refactor ann layer.

 private:

  /**
   * Apply pooling to the input and store the results.
   *
   * @param input The input to be apply the pooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Pooling(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    const size_t rStep = kW;
    const size_t cStep = kH;

    for (size_t j = 0, colidx = 0; j < output.n_cols; ++j, colidx += dH)
    {
      for (size_t i = 0, rowidx = 0; i < output.n_rows; ++i, rowidx += dW)
      {
        arma::mat subInput = input(
            arma::span(rowidx, rowidx + rStep - 1 - offset),
            arma::span(colidx, colidx + cStep - 1 - offset));

        output(i, j) = arma::mean(arma::mean(subInput));
      }
    }
  }

  /**
   * Apply unpooling to the input and store the results.
   *
   * @param input The input to be apply the unpooling rule.
   * @param output The pooled result.
   */
  template<typename eT>
  void Unpooling(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& error,
                 arma::Mat<eT>& output)
  {
    const size_t rStep = input.n_rows / error.n_rows - offset;
    const size_t cStep = input.n_cols / error.n_cols - offset;

    arma::Mat<eT> unpooledError;
    for (size_t j = 0; j < input.n_cols - cStep; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows - rStep; i += rStep)
      {
        const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1));

        unpooledError = arma::Mat<eT>(inputArea.n_rows, inputArea.n_cols);
        unpooledError.fill(error(i / rStep, j / cStep) / inputArea.n_elem);

        output(arma::span(i, i + rStep - 1 - offset),
            arma::span(j, j + cStep - 1 - offset)) += unpooledError;
      }
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

  //! Locally-stored input width.
  size_t inputWidth;

  //! Locally-stored input height.
  size_t inputHeight;

  //! Locally-stored output width.
  size_t outputWidth;

  //! Locally-stored output height.
  size_t outputHeight;

   //! Locally-stored reset parameter used to initialize the module once.
  bool reset;

  //! Rounding operation used.
  bool floor;

   //! If true use maximum a posteriori during the forward pass.
  bool deterministic;

  //! Locally-stored stored rounding offset.
  size_t offset;

  //! Locally-stored output parameter.
  arma::cube outputTemp;

  //! Locally-stored transformed input parameter.
  arma::cube inputTemp;

  //! Locally-stored transformed output parameter.
  arma::cube gTemp;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class MeanPooling


} // namespace ann
} // namespace mlpack

<<<<<<< HEAD
// Include implementation.
#include "mean_pooling_impl.hpp"

#endif
=======
#endif
>>>>>>> Refactor ann layer.
