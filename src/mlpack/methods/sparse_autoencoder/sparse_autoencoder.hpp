/**
 * @file sparse_autoencoder.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of sparse autoencoders.
 */
#ifndef __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP
#define __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/sparse_autoencoder/sparse_autoencoder_function.hpp>

namespace mlpack {
namespace nn {

/**
 * A sparse autoencoder is a neural network whose aim to learn compressed
 * representations of the data, typically for dimensionality reduction, with a
 * constraint on the activity of the neurons in the network. Sparse autoencoders
 * can be stacked together to learn a hierarchy of features, which provide a
 * better representation of the data for classification. This is a method used
 * in the recently developed field of deep learning. More technical details
 * about the model can be found on the following webpage:
 *
 * http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Data matrix.
 * const size_t vSize = 64; // Size of visible layer, depends on the data.
 * const size_t hSize = 25; // Size of hidden layer, depends on requirements.
 *
 * // Train the model using default options.
 * SparseAutoencoder encoder1(data, vSize, hSize);
 *
 * const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
 * const size_t numIterations = 100; // Maximum number of iterations.
 *
 * // Use an instantiated optimizer for the training.
 * SparseAutoencoderFunction saf(data, vSize, hSize);
 * L_BFGS<SparseAutoencoderFunction> optimizer(saf, numBasis, numIterations);
 * SparseAutoencoder<L_BFGS> encoder2(optimizer);
 *
 * arma::mat features1, features2; // Matrices for storing new representations.
 *
 * // Get new representations from the trained models.
 * encoder1.GetNewFeatures(data, features1);
 * encoder2.GetNewFeatures(data, features2);
 * @endcode
 *
 * This implementation allows the use of arbitrary mlpack optimizers via the
 * OptimizerType template parameter.
 *
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
 * @tparam OptimizerType The optimizer to use; by default this is L-BFGS.  Any
 *     mlpack optimizer can be used here.
 */
template<
  typename HiddenLayer = ann::SigmoidLayer<ann::LogisticFunction>,
  typename OutputLayer = HiddenLayer,
  template<typename> class OptimizerType = mlpack::optimization::L_BFGS
  >
class SparseAutoencoder
{
 public:
  using HidLayer = HiddenLayer;
  using OutLayer = OutputLayer;
 
  /**
   * Initialize the sparse autoencoder without performing training. The
   * Parameters(vector of w1,w2,b1,b2) will be initialize to zero.
   * lambda will be 0.0001, beta is 3 and rho is 0.01. Be sure to use
   * Train() before calling Predict(), otherwise the results may be
   * meaningless.
   *
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   */
  SparseAutoencoder(size_t visibleSize, size_t hiddenSize);

  /**
   * Construct the sparse autoencoder model with the given training data. This
   * will train the model. The parameters 'lambda', 'beta' and 'rho' can be set
   * optionally. Changing these parameters will have an effect on regularization
   * and sparsity of the model.
   *
   * @param data Input data with each column as one example.
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoder(const arma::mat& data,
                    size_t visibleSize,
                    size_t hiddenSize,
                    double lambda = 0.0001,
                    double beta = 3,
                    double rho = 0.01);

  /**
   * Construct the sparse autoencoder model with the given training data. This
   * will train the model. This overload takes an already instantiated optimizer
   * and uses it to train the model. The optimizer should hold an instantiated
   * SparseAutoencoderFunction object for the function to operate upon. This
   * option should be preferred when the optimizer options are to be changed.
   *
   * @param optimizer Instantiated optimizer with instantiated error function.
   */
  template<typename SparseAutoEncoderFunc>
  explicit SparseAutoencoder(OptimizerType<SparseAutoEncoderFunc>& optimizer);

  /**
   * Train the sparse autoencoder with the given training data.
   * @param data Input data with each column as one example.
   * @param Size of input vector expected at the hidden layer.
   * @return Objective value of the final point.
   */
  double Train(arma::mat const &data, size_t hiddenSize);

  /**
   * Train the sparse autoencoder model with the given optimizer.
   * The optimizer should hold an instantiated
   * SparseAutoencoderFunction object for the function to operate upon. This
   * option should be preferred when the optimizer options are to be changed.
   *
   * @param optimizer Instantiated optimizer with instantiated error function.
   * @return Objective value of the final point.
   */
  template<typename SparseAutoEncoderFunc>
  double Train(OptimizerType<SparseAutoEncoderFunc>& optimizer);

  /**
   * Transforms the provided data into the representation learned by the sparse
   * autoencoder. The function basically performs a feedforward computation
   * using the learned weights, and returns the hidden layer activations.
   *
   * @param data Matrix of the provided data.
   * @param features The hidden layer representation of the provided data.
   */
  void GetNewFeatures(const arma::mat& data, arma::mat& features) const;

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

  //! Sets size of the visible layer.
  void VisibleSize(size_t visible)
  {
    this->visibleSize = visible;
  }

  //! Gets size of the visible layer.
  size_t VisibleSize() const
  {
    return visibleSize;
  }

  //! Sets size of the hidden layer.
  void HiddenSize(size_t hidden)
  {
    this->m_ = hidden;
  }

  //! Gets the size of the hidden layer.
  size_t HiddenSize() const
  {
    return hiddenSize;
  }

  //! Sets the L2-regularization parameter.
  void Lambda(double l)
  {
    this->lambda = l;
  }

  //! Gets the L2-regularization parameter.
  double Lambda() const
  {
    return lambda;
  }

  //! Sets the KL divergence parameter.
  void Beta(double b)
  {
    this->beta = b;
  }

  //! Gets the KL divergence parameter.
  double Beta() const
  {
    return beta;
  }

  //! Sets the sparsity parameter.
  void Rho(double r)
  {
    this->rho = r;
  }

  //! For modification
  arma::mat& Parameters()
  {
    return parameters;
  }

  //! For access
  const arma::mat& Parameters() const
  {
    return parameters;
  }

  //! Gets the sparsity parameter.
  double Rho() const
  {
    return rho;
  }

  /**
   * Serialize the SparseAutoencoder
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using mlpack::data::CreateNVP;

    ar & CreateNVP(parameters, "parameters");
    ar & CreateNVP(visibleSize, "visibleSize");
    ar & CreateNVP(hiddenSize, "hiddenSize");
    ar & CreateNVP(lambda, "lambda");
    ar & CreateNVP(beta, "beta");
    ar & CreateNVP(rho, "rho");
  }

 private:  
  using SAEF = SparseAutoencoderFunction<HiddenLayer, OutputLayer>;

  //! Parameters after optimization.
  arma::mat parameters;
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
};

}; // namespace nn
}; // namespace mlpack

#include "sparse_autoencoder_impl.hpp"

#endif
