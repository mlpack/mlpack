/**
 * @file drop_connect_layer.hpp
 * @author Abhinav Chanda
 *
 * Definition of the DropConnectLayer class, which implements a regulaizer
 * that randomly sets weights to zero between two fully connected layers.
 * It prevents units from co-adapting and is a generalisation of dropout.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_DROPCONNECT_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_DROPCONNECT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dropconnect layer is a regularizer that randomly with probability ratio
 * sets weights to zero and scales the remaining weights by factor 1 /
 * (1 - ratio) between two fully connected layers. If rescale is true the input 
 * is scaled with 1 / (1-p) when deterministic is false. In the deterministic mode
 * (during testing), the layer just scales the output.
 *
 * Note: During training you should set deterministic to false and during
 * testing you should set deterministic to true.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Wan2013,
 *   author  = {Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, Rob Fergus},
 *   title   = {Regularization of Neural Networks using DropConnect},
 *   journal = {JMLR},
 *   volume  = {28},
 *   year    = {2013},
 * }
 * @endcode
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
class DropConnectLayer
{
 public:
  /**
   * Create the DropConnectLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  DropConnectLayer(const size_t inSize,
                   const size_t outSize,
                   const double ratio = 0.5,
                   const bool rescale = true) :
      inSize(inSize),
      outSize(outSize),
      ratio(ratio),
      scale(1.0 / (1.0 - ratio)),
      rescale(rescale)
  {
    weights.set_size(outSize, inSize);
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
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      if (!rescale)
      {
        output = weights * input;
      }
      else
      {
        output = weights * scale * input;
      }
    }
    else
    {
      // Scale with input / (1 - ratio) and set values to zero with probability
      // ratio.
      mask = arma::randu<arma::Mat<eT> >(weights.n_rows, weights.n_cols);
      mask.transform( [&](double val) { return (val > ratio); } );
      output = (weights % mask) * scale * input;
    }
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Mat<eT>& output)
  {
    // The dropout mask will not be multiplied in the deterministic mode
    // (during testing).
    if (deterministic)
    {
      if (!rescale)
      {
        output = weights * input;
      }
      else
      {
        output = weights * scale * input;
      }
    }
    else
    {
      arma::Mat<eT> data(input.n_elem, 1);

      for (size_t s = 0, c = 0; s < input.n_slices / data.n_cols; s++)
      {
        for (size_t i = 0; i < data.n_cols; i++, c++)
        {
          data.col(i).subvec(s * input.n_rows * input.n_cols, (s + 1) *
            input.n_rows * input.n_cols - 1) = arma::vectorise(input.slice(c));
        }
      }
      // Scale with input / (1 - ratio) and set values to zero with probability
      // ratio.
      mask = arma::randu<arma::Mat<eT> >(weights.n_rows, weights.n_cols);
      mask.transform( [&](double val) { return (val > ratio); } );
      output = (weights % mask) * scale * data;
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
  template<typename InputType, typename eT>
  void Backward(const InputType& /* unused */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    g = ((weights % mask) * scale).t() * gy;
  }
  

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT, typename GradientDataType>
  void Gradient(const arma::Mat<eT>& d, GradientDataType& g)
  {
    GradientDelta(inputParameter, d, g);
  }

  //! Get the weights.
  OutputDataType& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

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

  //! Get the gradient.
  OutputDataType& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  //! The value of the deterministic parameter.
  bool Deterministic() const { return deterministic; }
  //! Modify the value of the deterministic parameter.
  bool& Deterministic() { return deterministic; }

  //! The probability of setting a value to zero.
  double Ratio() const { return ratio; }

  //! Modify the probability of setting a value to zero.
  void Ratio(const double r)
  {
    ratio = r;
    scale = 1.0 / (1.0 - ratio);
  }

  //! The value of the rescale parameter.
  bool Rescale() const {return rescale; }
  //! Modify the value of the rescale parameter.
  bool& Rescale() {return rescale; }

  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(ratio, "ratio");
    ar & data::CreateNVP(rescale, "rescale");
    ar & data::CreateNVP(weights, "weights");
  }

 private:
   /*
   * Calculate the gradient using the output delta (3rd order tensor) and the
   * input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    arma::Mat<eT> data = arma::Mat<eT>(d.n_cols,
        input.n_elem / d.n_cols);

    for (size_t s = 0, c = 0; s < input.n_slices /
        data.n_rows; s++)
    {
      for (size_t i = 0; i < data.n_rows; i++, c++)
      {
        data.row(i).subvec(s * input.n_rows *
            input.n_cols, (s + 1) *
            input.n_rows *
            input.n_cols - 1) = arma::vectorise(
                input.slice(c), 1);
      }
    }

    g.slice(0) = d * data / d.n_cols;
  }

  /*
   * Calculate the gradient (3rd order tensor) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& /* input unused */,
                     const arma::Mat<eT>& d,
                     arma::Cube<eT>& g)
  {
    g = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(d, g.slice(0));
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (3rd order tensor).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Cube<eT>& /* input unused */,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    arma::Cube<eT> grad = arma::Cube<eT>(weights.n_rows, weights.n_cols, 1);
    Gradient(d, grad);
    g = grad.slice(0);
  }

  /*
   * Calculate the gradient (dense matrix) using the output delta
   * (dense matrix) and the input activation (dense matrix).
   *
   * @param input The input parameter used for calculating the gradient.
   * @param d The output delta.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void GradientDelta(const arma::Mat<eT>& input,
                     const arma::Mat<eT>& d,
                     arma::Mat<eT>& g)
  {
    g = d * input.t();
  }

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored mask object.
  OutputDataType mask;

  //! The probability of setting a value to zero.
  double ratio;

  //! The scale fraction.
  double scale;

  //! If true dropout and scaling is disabled, see notes above.
  bool deterministic;

  //! If true the input is rescaled when deterministic is False.
  bool rescale;
}; // class DropConnectLayer

/**
 * Mapping layer to map between 3rd order tensors and dense matrices.
 */
template <
    typename InputDataType = arma::cube,
    typename OutputDataType = arma::mat
>
using DropConnectLinearMappingLayer = DropConnectLayer<InputDataType, OutputDataType>;

//! Layer traits for the dropconnect layer.
template<
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<DropConnectLayer<InputDataType, OutputDataType> >
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
