/**
 * @file pooling_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the PoolingLayer class, which attaches various pooling
 * functions to the embedding layer.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_POOLING_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_POOLING_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/pooling_rules/mean_pooling.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the pooling layer. The pooling layer works as a metaclass
 * which attaches various functions to the embedding layer.
 *
 * @tparam PoolingRule Pooling function used for the embedding layer.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename PoolingRule = MeanPooling,
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::cube
>
class PoolingLayer
{
 public:
  /**
   * Create the PoolingLayer object using the specified number of units.
   *
   * @param kSize Size of the pooling window.
   * @param pooling The pooling strategy.
   */
  PoolingLayer(const size_t kSize, PoolingRule pooling = PoolingRule()) :
      kSize(kSize), pooling(pooling)
  {
    // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {
    Pooling(input, output);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = arma::zeros<arma::Cube<eT> >(input.n_rows / kSize,
                            input.n_cols / kSize, input.n_slices);

    for (size_t s = 0; s < input.n_slices; s++)
      Pooling(input.slice(s), output.slice(s));
  }

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
  void Backward(const arma::Cube<eT>& /* unused */,
                const arma::Cube<eT>& gy,
                arma::Cube<eT>& g)
  {
    g = arma::zeros<arma::Cube<eT> >(inputParameter.n_rows,
        inputParameter.n_cols, inputParameter.n_slices);

    for (size_t s = 0; s < gy.n_slices; s++)
    {
      Unpooling(inputParameter.slice(s), gy.slice(s), g.slice(s));
    }
  }

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
  void Backward(const arma::Cube<eT>& /* unused */,
                const arma::Mat<eT>& gy,
                arma::Cube<eT>& g)
  {
    // Generate a cube from the error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(outputParameter.n_rows,
        outputParameter.n_cols, outputParameter.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
    {
      for (size_t i = 0; i < gy.n_cols; i++)
      {
        arma::Col<eT> temp = gy.col(i).subvec(
            j * outputParameter.n_rows * outputParameter.n_cols,
            (j + 1) * outputParameter.n_rows * outputParameter.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            outputParameter.n_rows, outputParameter.n_cols);
      }
    }

    Backward(inputParameter, mappedError, g);
  }

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  InputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  InputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(kSize, "kSize");
    ar & data::CreateNVP(pooling, "pooling");
  }

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

    const size_t rStep = kSize;
    const size_t cStep = kSize;

    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        output(i / rStep, j / cStep) += pooling.Pooling(
            input(arma::span(i, i + rStep - 1), arma::span(j, j + cStep - 1)));
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
    const size_t rStep = input.n_rows / error.n_rows;
    const size_t cStep = input.n_cols / error.n_cols;

    arma::Mat<eT> unpooledError;
    for (size_t j = 0; j < input.n_cols; j += cStep)
    {
      for (size_t i = 0; i < input.n_rows; i += rStep)
      {
        const arma::Mat<eT>& inputArea = input(arma::span(i, i + rStep - 1),
                                               arma::span(j, j + cStep - 1));

        pooling.Unpooling(inputArea, error(i / rStep, j / cStep),
            unpooledError);

        output(arma::span(i, i + rStep - 1),
            arma::span(j, j + cStep - 1)) += unpooledError;
      }
    }
  }

  //! Locally-stored size of the pooling window.
  size_t kSize;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pooling strategy.
  PoolingRule pooling;
}; // class PoolingLayer

//! Layer traits for the pooling layer.
template<
    typename PoolingRule,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<PoolingLayer<PoolingRule, InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};


} // namespace ann
} // namespace mlpack

#endif
