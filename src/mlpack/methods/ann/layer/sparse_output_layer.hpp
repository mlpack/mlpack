/**
 * @file sparse_output_layer.hpp
 * @author Tham Ngap Wei
 *
 * This is the fourth layer of sparse autoencoder.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_SPARSE_OUTPUT_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_SPARSE_OUTPUT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the SparseOutputLayer class. The SparseOutputLayer class
 * represents  the fourth layer of the sparse autoencoder.
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
class SparseOutputLayer
{
 public:
  /**
   * Create the SparseLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   */
  SparseOutputLayer(const size_t inSize,
                    const size_t outSize,
                    const double lambda = 0.0001,
                    const double beta = 3,
                    const double rho = 0.01) :
    inSize(inSize),
    outSize(outSize),
    lambda(lambda),
    beta(beta),
    rho(rho)
  {
    weights.set_size(outSize, inSize);
  }

  SparseOutputLayer(SparseOutputLayer &&layer) noexcept
  {
    *this = std::move(layer);
  }

  SparseOutputLayer& operator=(SparseOutputLayer &&layer) noexcept
  {
    beta = layer.beta;
    rho = layer.rho;
    lambda = layer.lambda;
    inSize = layer.inSize;
    outSize = layer.outSize;
    weights.swap(layer.weights);

    return *this;
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
    output = weights * input;
    // Average activations of the hidden layer.
    rhoCap = arma::sum(input, 1) / static_cast<double>(input.n_cols);    
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
  void Backward(const InputType& input,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    const arma::mat klDivGrad = beta * (-(rho / rhoCap) + (1 - rho) /
          (1 - rhoCap));

    // NOTE: if the armadillo version high enough, find_nonfinite can prevents
    // overflow value:
    // klDivGrad.elem(arma::find_nonfinite(klDivGrad)).zeros();
    g = weights.t() * gy +
        arma::repmat(klDivGrad, 1, input.n_cols);
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& d, arma::Mat<eT>& g)
  {        
    g = d * inputParameter.t() /
        static_cast<typename InputDataType::value_type>(inputParameter.n_cols) +
        lambda * weights;    
  }
  
  //! Sets the KL divergence parameter.
  void Beta(const double b)
  {
    beta = b;
  }

  //! Gets the KL divergence parameter.
  double Beta() const
  {
    return beta;
  }

  //! Sets the sparsity parameter.
  void Rho(const double r)
  {
    rho = r;
  }

  //! Gets the sparsity parameter.
  double Rho() const
  {
    return rho;
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the RhoCap.
  OutputDataType const& RhoCap() const { return rhoCap; }
  //! Modify the RhoCap.
  OutputDataType& RhoCap() { return rhoCap; }

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

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;
  
  //! L2-regularization parameter.
  double lambda;

  //! KL divergence parameter.
  double beta;
  
  //! Sparsity parameter.
  double rho;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Average activations of the hidden layer.
  OutputDataType rhoCap;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
}; // class SparseOutputLayer

//! Layer traits for the SparseOutputLayer.
template<typename InputDataType, typename OutputDataType
    >
class LayerTraits<SparseOutputLayer<InputDataType, OutputDataType> >
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
