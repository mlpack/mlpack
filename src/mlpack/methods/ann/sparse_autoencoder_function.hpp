#ifndef __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP

#include <mlpack/core.hpp>

#include <type_traits>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
 *   using FSigmoidLayer = ann::SigmoidLayer<ann::LazyLogisticFunction>;
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
 * @tparam HiddenLayer the layer type of the HiddenLayer
 * @tparam OutputLayer the layer type of the HiddenLayer
 * @tparam Greedy If it is true_type, the function Gradient will\n
 * recalculate part of the calculation done in the function Evaluate\n
 * and vice versa.By default it is false_type
 */
template<typename HiddenLayer, typename OutputLayer,
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
    const arma::mat InitializeWeights();

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
    const arma::mat& GetInitialPoint() const { return initialPoint; }

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
        hiddenLayerFunc.Forward(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
                                arma::repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
                                hiddenLayer);

        outputLayerFunc.Forward(parameters.submat(l1, 0, l3 - 1, l2 - 1).t() * hiddenLayer +
                                arma::repmat(parameters.submat(l3, 0, l3, l2 - 1).t(), 1, data.n_cols),
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

template<typename HiddenLayer, typename OutputLayer, typename Greedy>
SparseAutoencoderFunction<HiddenLayer, OutputLayer, Greedy>::
SparseAutoencoderFunction(const arma::mat& data,
                          const size_t visibleSize,
                          const size_t hiddenSize,
                          const double lambda,
                          const double beta,
                          const double rho) :
    data(data),
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    lambda(lambda),
    beta(beta),
    rho(rho)
{
    // Initialize the parameters to suitable values.
    initialPoint = InitializeWeights();
}

/** Initializes the parameter weights if the initial point is not passed to the
  * constructor. The weights w1, w2 are initialized to randomly in the range
  * [-r, r] where 'r' is decided using the sizes of the visible and hidden
  * layers. The biases b1, b2 are initialized to 0.
  */
template<typename HiddenLayer, typename OutputLayer, typename Greedy>
const arma::mat SparseAutoencoderFunction<HiddenLayer, OutputLayer, Greedy>::
InitializeWeights()
{
    // The module uses a matrix to store the parameters, its structure looks like:
    //          vSize   1
    //       |        |  |
    //  hSize|   w1   |b1|
    //       |________|__|
    //       |        |  |
    //  hSize|   w2'  |  |
    //       |________|__|
    //      1|   b2'  |  |
    //
    // There are (hiddenSize + 1) empty cells in the matrix, but it is small
    // compared to the matrix size. The above structure allows for smooth matrix
    // operations without making the code too ugly.

    // Initialize w1 and w2 to random values in the range [0, 1], then set b1 and
    // b2 to 0.
    arma::mat parameters;
    parameters.randu(2 * hiddenSize + 1, visibleSize + 1);
    parameters.row(2 * hiddenSize).zeros();
    parameters.col(visibleSize).zeros();

    // Decide the parameter 'r' depending on the size of the visible and hidden
    // layers. The formula used is r = sqrt(6) / sqrt(vSize + hSize + 1).
    const double range = sqrt(6) / sqrt(visibleSize + hiddenSize + 1);

    //Shift range of w1 and w2 values from [0, 1] to [-r, r].
    parameters.submat(0, 0, 2 * hiddenSize - 1, visibleSize - 1) = 2 * range *
            (parameters.submat(0, 0, 2 * hiddenSize - 1, visibleSize - 1) - 0.5);

    return parameters;
}

/** Evaluates the objective function given the parameters.
  */
template<typename HiddenLayer, typename OutputLayer, typename Greedy>
double SparseAutoencoderFunction<HiddenLayer, OutputLayer, Greedy>::
Evaluate(const arma::mat& parameters)
{
    // The objective function is the average squared reconstruction error of the
    // network. w1 and b1 are the weights and biases associated with the hidden
    // layer, whereas w2 and b2 are associated with the output layer.
    // f(w1,w2,b1,b2) = sum((data - sigmoid(w2*sigmoid(w1data + b1) + b2))^2) / 2m
    // 'm' is the number of training examples.
    // The cost also takes into account the regularization and KL divergence terms
    // to control the parameter weights and sparsity of the model respectively.

    // Compute the limits for the parameters w1, w2, b1 and b2.
    const size_t l1 = hiddenSize;
    const size_t l2 = visibleSize;
    const size_t l3 = 2 * hiddenSize;

    EvalParams(parameters, l1, l2, l3, std::true_type());

    // Calculate squared L2-norms of w1 and w2.
    const double wL2SquaredNorm = arma::accu(parameters.submat(0, 0, l3 - 1, l2 - 1) %
                                             parameters.submat(0, 0, l3 - 1, l2 - 1));

    // Calculate the reconstruction error, the regularization cost and the KL
    // divergence cost terms. 'sumOfSquaresError' is the average squared l2-norm
    // of the reconstructed data difference. 'weightDecay' is the squared l2-norm
    // of the weights w1 and w2. 'klDivergence' is the cost of the hidden layer
    // activations not being low. It is given by the following formula:
    // KL = sum_over_hSize(rho*log(rho/rhoCaq) + (1-rho)*log((1-rho)/(1-rhoCap)))
    const double sumOfSquaresError = 0.5 * arma::accu(diff % diff) / data.n_cols;
    const double weightDecay = 0.5 * lambda * wL2SquaredNorm;
    const double klDivergence = beta * arma::accu(rho * arma::log(rho / rhoCap) + (1 - rho) *
                                                  arma::log((1 - rho) / (1 - rhoCap)));

    // The cost is the sum of the terms calculated above.
    return sumOfSquaresError + weightDecay + klDivergence;
}

/** Calculates and stores the gradient values given a set of parameters.
  */
template<typename HiddenLayer, typename OutputLayer, typename Greedy>
void SparseAutoencoderFunction<HiddenLayer, OutputLayer, Greedy>::
Gradient(const arma::mat& parameters,
         arma::mat& gradient)
{
    // Performs a feedforward pass of the neural network, and computes the
    // activations of the output layer as in the Evaluate() method. It uses the
    // Backpropagation algorithm to calculate the delta values at each layer,
    // except for the input layer. The delta values are then used with input layer
    // and hidden layer activations to get the parameter gradients.

    // Compute the limits for the parameters w1, w2, b1 and b2.
    const size_t l1 = hiddenSize;
    const size_t l2 = visibleSize;
    const size_t l3 = 2 * hiddenSize;

    EvalParams(parameters, l1, l2, l3, Greedy());

    // The delta vector for the output layer is given by diff * f'(z), where z is
    // the preactivation and f is the activation function. The derivative of the
    // sigmoid function turns out to be f(z) * (1 - f(z)). For every other layer
    // in the neural network which comes before the output layer, the delta values
    // are given del_n = w_n' * del_(n+1) * f'(z_n). Since our cost function also
    // includes the KL divergence term, we adjust for that in the formula below.
    arma::mat delOut;
    outputLayerFunc.Backward(outputLayer, diff, delOut);
    arma::mat const klDivGrad = beta * (-(rho / rhoCap) + (1 - rho) / (1 - rhoCap));
    diff2 = parameters.submat(l1, 0, l3 - 1, l2 - 1) * delOut +
            arma::repmat(klDivGrad, 1, data.n_cols);
    arma::mat delHid;
    hiddenLayerFunc.Backward(hiddenLayer,
                             diff2,
                             delHid);

    gradient.zeros(2 * hiddenSize + 1, visibleSize + 1);

    // Compute the gradient values using the activations and the delta values. The
    // formula also accounts for the regularization terms in the objective.
    // function.
    // cast to double, this could prevent warning message
    double const sampleSize = static_cast<double>(data.n_cols);
    gradient.submat(0, 0, l1 - 1, l2 - 1) = delHid * data.t() / sampleSize +
            lambda * parameters.submat(0, 0, l1 - 1, l2 - 1);
    gradient.submat(l1, 0, l3 - 1, l2 - 1) =
            (delOut * hiddenLayer.t() / sampleSize +
             lambda * parameters.submat(l1, 0, l3 - 1, l2 - 1).t()).t();
    gradient.submat(0, l2, l1 - 1, l2) = arma::sum(delHid, 1) / sampleSize;
    gradient.submat(l3, 0, l3, l2 - 1) = (arma::sum(delOut, 1) / sampleSize).t();
}; // class SparseAutoencoderFunction

}; // namespace ann
}; // namespace mlpack

#endif
