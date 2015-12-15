#ifndef __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

#include <type_traits>

namespace mlpack {
namespace nn /** Artificial Neural Network. */ {

/**
 * This is a class for the sparse autoencoder objective function. It can be used
 * to create learning models like self-taught learning, stacked autoencoders,
 * conditional random fields (CRFs), and so forth.
 * @code
 * #include <mlpack/methods/ann/sparse_autoencoder_function.hpp>
 * #include <mlpack/methods/ann/activation_functions/lazy_logistic_function.hpp>
 *
 * #include <mlpack/core.hpp>
 * #include <mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp>
 * #include <mlpack/methods/ann/layer/base_layer.hpp>
 *
 * #include <iostream>
 *
 * int main()
 *  {
 *   using namespace mlpack;
 *   using namespace arma;
 *
 *   using FSigmoidLayer = ann::SigmoidLayer<ann::LogisticFunction>;
 *
 *   using SAEF = ann::SparseAutoencoderFunction<FSigmoidLayer, FSigmoidLayer>;
 *   //If you want to compare the performance with original class, replace
 *   //SAEF with following definition
 *   //using SAEF = nn::SparseAutoencoderFunction;
 *
 *   size_t const Features = 16*16;
 *   arma::mat data = randu<mat>(Features, 10000);
 *
 *   SAEF encoderFunction(data, Features, Features / 2);
 *   const size_t numIterations = 100; // Maximum number of iterations.
 *   const size_t numBasis = 10;
 *   optimization::L_BFGS<SAEF> optimizer(encoderFunction, numBasis, numIterations);
 *
 *   arma::mat parameters = encoderFunction.GetInitialPoint();
 *
 *   // Train the model.
 *   Timer::Start("sparse_autoencoder_optimization");
 *   const double out = optimizer.Optimize(parameters);
 *   Timer::Stop("sparse_autoencoder_optimization");
 *   std::cout<<"spend time : "<<
 *             Timer::Get("sparse_autoencoder_optimization").tv_sec<<"\n";
 *
 *   std::cout << "SparseAutoencoder::SparseAutoencoder(): final objective of "
 *             << "trained model is " << out << "." << std::endl;
 *}
 * @endcode
 * @tparam HiddenLayer The layer type of the hidden layer, this type must
 * provide functions "Forward(const InputType& input, OutputType& output)"
 * and "Backward(const DataType& input, const DataType& gy, DataType& g)"
 * the InputType, OutpuType, DataType must be able to accept arma::mat.
 *
 * @tparam OutputLayer The layer type of the output, this type must
 * provide functions "Forward(const InputType& input, OutputType& output)"
 * and "Backward(const DataType& input, const DataType& gy, DataType& g)"
 * the InputType, OutpuType, DataType must be able to accept arma::mat.
 *
 * @tparam Greedy If it is true_type, the function Gradient will\n
 * recalculate part of the calculation done in the function Evaluate\n
 * and vice versa.By default it is false_type
 */
template<typename HiddenLayer = ann::SigmoidLayer<nn::LogisticFunction>,
         typename OutputLayer = HiddenLayer,
         typename Greedy = std::false_type>
class SparseAutoencoderFunction
{
 public:
  /**
   * Construct the sparse autoencoder objective function with the given
   * parameters.
   *
   * @param data The data matrix.
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoderFunction(const arma::mat& data,
                            const size_t visibleSize,
                            const size_t hiddenSize,
                            const double lambda = 0.0001,
                            const double beta = 3,
                            const double rho = 0.01); 

  //! Initializes the parameters of the model to suitable values.
  const arma::mat& InitializeWeights();

  /**
   * Initializes the parameters of the model to suitable values.
   * @param weights weights want to initialize
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   */
  static void InitializeWeights(arma::mat &weights,
                                const size_t visibleSize,
                                const size_t hiddenSize);

  /**
   * Initializes the parameters of the model to suitable values.
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   * @param weights after initialize
   */
  static arma::mat InitializeWeights(const size_t visibleSize,
                                     const size_t hiddenSize);

  /**
   * Evaluates the objective function of the sparse autoencoder model using the
   * given parameters. The cost function has terms for the reconstruction
   * error, regularization cost and the sparsity cost. The objective function
   * takes a low value when the model is able to reconstruct the data well
   * using weights which are low in value and when the average activations of
   * neurons in the hidden layers agrees well with the sparsity parameter 'rho'.
   *
   * @param parameters Current values of the model parameters.
   */
  double Evaluate(const arma::mat& parameters);

  /**
   * Evaluates the gradient values of the objective function given the current
   * set of parameters. The function performs a feedforward pass and computes
   * the error in reconstructing the data points. It then uses the
   * backpropagation algorithm to compute the gradient values.
   *
   * @param parameters Current values of the model parameters.
   * @param gradient Matrix where gradient values will be stored.
   */
  void Gradient(const arma::mat& parameters, arma::mat& gradient);



  /**
   * Returns the elementwise sigmoid of the passed matrix, where the sigmoid
   * function of a real number 'x' is [1 / (1 + exp(-x))].
   *
   * @param x Matrix of real values for which we require the sigmoid activation.
   */
  void Sigmoid(const arma::mat& x, arma::mat& output) const
  {
    output = (1.0 / (1 + arma::exp(-x)));
  }

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const {
    return initialPoint;
  }

  //! Sets size of the visible layer.
  void VisibleSize(const size_t visible)
  {
    this->visibleSize = visible;
  }

  //! Gets size of the visible layer.
  size_t VisibleSize() const
  {
    return visibleSize;
  }

  //! Sets size of the hidden layer.
  void HiddenSize(const size_t hidden)
  {
    this->hiddenSize = hidden;
  }

  //! Gets the size of the hidden layer.
  size_t HiddenSize() const
  {
    return hiddenSize;
  }

  //! Sets the L2-regularization parameter.
  void Lambda(const double l)
  {
    this->lambda = l;
  }

  //! Gets the L2-regularization parameter.
  double Lambda() const
  {
    return lambda;
  }

  //! Sets the KL divergence parameter.
  void Beta(const double b)
  {
    this->beta = b;
  }

  //! Gets the KL divergence parameter.
  double Beta() const
  {
    return beta;
  }

  //! Sets the sparsity parameter.
  void Rho(const double r)
  {
    this->rho = r;
  }

  //! Gets the sparsity parameter.
  double Rho() const
  {
    return rho;
  }

 private:
  void EvalParams(arma::mat const &parameters, size_t l1, size_t l2, size_t l3,
                  std::true_type /* unused */)
  {
    // w1, w2, b1 and b2 are not extracted separately, 'parameters' is directly
    // used in their place to avoid copying data. The following representations
    // are used:
    // w1 <- parameters.submat(0, 0, l1-1, l2-1)
    // w2 <- parameters.submat(l1, 0, l3-1, l2-1).t()
    // b1 <- parameters.submat(0, l2, l1-1, l2)
    // b2 <- parameters.submat(l3, 0, l3, l2-1).t()

    // Compute activations of the hidden and output layers.
    arma::mat tempInput = parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
                          arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols);
    hiddenLayerFunc.Forward(tempInput,
                            hiddenLayer);

    tempInput = parameters.submat(l1, 0, l3 - 1, l2 - 1).t() * hiddenLayer +
                arma::repmat(parameters.submat(l3, 0, l3, l2 - 1).t(), 1, data.n_cols);
    outputLayerFunc.Forward(tempInput,
                            outputLayer);

    // Average activations of the hidden layer.
    rhoCap = arma::sum(hiddenLayer, 1) / static_cast<double>(data.n_cols);
    // Difference between the reconstructed data and the original data.
    diff = outputLayer - data;
  }

  void EvalParams(arma::mat const& /* unused */, size_t /* unused */,
                  size_t /* unused */, size_t /* unused */,
                  std::false_type)
  {

  }

  //! The matrix of data points.
  const arma::mat& data;
  //! Intial parameter vector.
  arma::mat initialPoint;
  //! Size of the visible layer.
  size_t visibleSize;
  //! Size of the hidden layer.
  size_t hiddenSize;
  //! L2-regularization parameter.
  double lambda;
  //! KL divergence parameter.
  double beta;
  //! Sparsity parameter.
  double rho;
  //!activation of hidden layer
  arma::mat hiddenLayer;
  //!activation of output layer
  arma::mat outputLayer;
  //!Average activations of the hidden layer.
  arma::mat rhoCap;
  //!Difference between the reconstructed data and the original data.
  arma::mat diff;
  //!Difference(error) between the output layer and the hidden layer.
  arma::mat diff2;

  HiddenLayer hiddenLayerFunc;
  OutputLayer outputLayerFunc;
};

} // namespace nn
} // namespace mlpack

#include "sparse_autoencoder_function_impl.hpp"

#endif
