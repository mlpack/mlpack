/**
 * @file leaky_relu_layer.hpp
 * @author Dhawal Arora
 *
 * Implementation of Leaky_ReLU activation function by 
 * Maas, Andrew L, Hannun, Awni Y, and Ng, An-drew Y. Rectifier nonlinearities improve neural net-
 * work acoustic models. In ICML, volume 30, 2013.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_LEAKYRELU_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_LEAKYRELU_LAYER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The LeakyReLU activation function, defined by
 *
 * @f{eqnarray*}{
 * f(x) &=& \max(x, alpha*x) \\
 * f'(x) &=& \left\{
 *   \begin{array}{lr}
 *     1 & : x > 0 \\
 *     alpha & : x \le 0
 *   \end{array}
 * \right.
 * @f}
 */

template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>

class LeakyReLULayer
{
 public:

  /**
   * Constructor. Default value for alpha is set to 0.03
   *
   */

  LeakyReLULayer(const double a = 0.03) : alpha(a)
  {
     // Nothing to do here.
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   * 
   * @param x Input data used for evaluating the specified function. This is for just one input value. 
   * @return f(x) The activation value for the input.  
   */

  double Forward(const double x)
  {
    return fn(x);
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
    fn(input, output);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation. This function is for just a single input. 
   * @param gy The backpropagated error.
   * @return The calculated gradient.
   */

  double Backward(const double input,
                  const double gy)
  {
    return gy * deriv(input);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType>
  void Backward(const DataType& input,
                const DataType& gy,
                DataType& g)
  {
    DataType derivative;
    deriv(input, derivative);
    g = gy % derivative;
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards through f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Cube<eT>& input,
                const arma::Mat<eT>& gy,
                arma::Cube<eT>& g)
  {
    // Generate a cube using the backpropagated error matrix.
    arma::Cube<eT> mappedError = arma::zeros<arma::cube>(input.n_rows,
        input.n_cols, input.n_slices);

    for (size_t s = 0, j = 0; s < mappedError.n_slices; s+= gy.n_cols, j++)
    {
      for (size_t i = 0; i < gy.n_cols; i++)
      {
        arma::Col<eT> temp = gy.col(i).subvec(
            j * input.n_rows * input.n_cols,
            (j + 1) * input.n_rows * input.n_cols - 1);

        mappedError.slice(s + i) = arma::Mat<eT>(temp.memptr(),
            input.n_rows, input.n_cols);
      }
    }

    arma::Cube<eT> derivative;
    deriv(input, derivative);
    g = mappedError % derivative;
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

  //! Get the Leakyness Parameter.
  double const& Alpha() const { return alpha; }
  //! Modify the Leakyness Parameter.
  double& Alpha() { return alpha; }

  
  /**
   * Serialize the layer.
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */)
  {
    /* Nothing to do here */
  }

 private:

  /**
   * Computes the leaky ReLU function where 0 <alpha< 1
   *
   * @param x Input data.
   * @return f(x).
   */
  double fn(const double x)
  {
    return std::max(x, alpha * x);
  }

  /**
   * Computes the Leaky ReLU function using a dense matrix as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    y = arma::max(x, alpha * x);
  }

  /**
   * Computes the LeakyReLU function using a 3rd-order tensor as input.
   *
   * @param x Input data.
   * @param y The resulting output activation.
   */
  template<typename eT>
  void fn(const arma::Cube<eT>& x, arma::Cube<eT>& y)
  {
    y = x;
    for (size_t s = 0; s < x.n_slices; s++)
      fn(x.slice(s), y.slice(s));
  }

  /**
   * Computes the first derivative of the LeakyReLU function.
   *
   * @param x Input data.
   * @return f'(x)
   */
  double deriv(const double x)
  {
    return (x >= 0) ? 1 : alpha;
  }

  /**
   * Computes the first derivative of the LeakyReLU function.
   *
   * @param y Input activations.
   * @param x The resulting derivatives.
   */

  template<typename InputType, typename OutputType>
  void deriv(const InputType& x, OutputType& y)
  {
    y = x;

    for (size_t i = 0; i < x.n_elem; i++)
      y(i) = deriv(x(i));
  }



  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Leakyness Parameter in the range 0 <alpha< 1
  double alpha;

}; // class LeakyReLULayer

} // namespace ann
} // namespace mlpack

#endif