#include <mlpack/core.hpp>

#include <algorithm>
#include <type_traits>

namespace mlpack{
namespace nn{

/**
 * Fine tune deep network like StackAutoencoder
 *
 *@code
 * //assume following inputs and params already pretrain
 * arma::mat sae1Input;
 * arma::mat sae2Input;
 * arma::mat softmaxInput;
 *
 * arma::mat sae1Params;
 * arma::mat sae2Params;
 * arma::mat softmaxParams;
 *
 * using namespace mlpack;
 *
 * std::vector<arma::mat*> inputs{&sae1Input, &sae2Input,
 * &softmaxInput};
 * std::vector<arma::mat*> params{&sae1Params, &sae2Params,
 * &softmaxParams};
 *
 * regression::SoftmaxRegressionFunction smFunction(softmaxInput, labels, 2);
 *
 * using FineTuneFunction = nn::FineTuneFunction<regression::SoftmaxRegressionFunction,
 * nn::SoftmaxFineTune>;
 * FineTuneFunction finetune(inputs, params, smFunction);
 *
 * size_t numBasis = 5;
 * size_t numIteration = 400;
 *
 * mlpack::optimization::L_BFGS<FineTune>
 * optimizer(finetune, numBasis, numIteration);
 *
 * arma::mat fineTuneParameters;
 * optimizer.Optimize(fineTuneParameters);
 * finetune.UpdateParameters(fineTuneParameters);
 *
 *@endcode
 *@tparam OutputLayerType types of the output layer, must implement three functions.
 * Gradient(const arma::mat& parameters, arma::mat& gradient);
 * double Evaluate(const arma::mat& parameters);
 * arma::mat& GetInitialPoint();
 *@tparam FineTuneGradient Functor for calculating the last gradient, it should implement two functions
 * - template<typename T> Gradient(arma::mat const&, arma::mat const&, T const&, arma::mat&)
 * - Deriv(arma::mat const&, arma::mat&);
 */
template<typename OutputLayerType,
         typename FineTuneGradient>
class FineTuneFunction
{
public:        
    /**
     * Construct the class with given data
     * @param input The input data of the LayerTypes and OutputLayerType
     * @param parameters The parameters of the LayerTypes and OutputLayerType
     * @param layerTypes The type(must be tuple) of the Layer(by now only support SparseAutoencoder)
     * @param outLayerType The type of the last layer(ex : softmax)
     */
    FineTuneFunction(std::vector<arma::mat*> &input,
                     std::vector<arma::mat*> &parameters,
                     OutputLayerType &outLayerType);

    /**
     * Evaluates the objective function of the networks using the
     * given parameters.
     * @param parameters Current values of the model parameters.
     */
    double Evaluate(const arma::mat& parameters);    

    /**
     * Evaluates the gradient values of the objective function given the current
     * set of parameters. The function performs a feedforward pass and computes
     * the error in reconstructing the data points. It then uses the
     * backpropagation algorithm to compute the gradient values.
     * @param parameters Current values of the model parameters.
     * @param gradient Matrix where gradient values will be stored.
     */
    void Gradient(const arma::mat& parameters, arma::mat& gradient);

    //! Return initial point
    const arma::mat& GetInitialPoint() const
    {
        return initialPoint;
    }

    //! Generate the initial point for the optimization.
    void InitializeWeights();

    /**
     * Update the parameters of model with finetune parameters
     * @param parameters the parameters after finetune
     */
    void UpdateParameters(arma::mat const &parameters);    

private:
    /**
     *  Feed the two dimension parameters into one dimension parameters.
     *  The layout is [w1 b1 w2 b2.....wn bn softmax_params].
     *  If the w1 = [0 1; 2 3], after flatten, it will become
     *  [0 1 2 3]
     */
    void FlattenLayerParams();    

    void UpdateInputData(arma::mat const &parameters);    

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

    size_t HiddenSize(arma::mat const &parameters) const
    {
        return (parameters.n_rows - 1) / 2;
    }

    size_t VisibleSize(arma::mat const &parameters) const
    {
        return parameters.n_cols - 1;
    }

    /**
     * Calculate all of the encoder size of the sparse autoencoder
     * @return Total size of the encoder of sparse autoencoder
     */
    size_t TotalEncoderSize() const;    

    /**
     * Get the number of elements of w1 of the parameters
     * @param params the parameters
     * @return element size of the weights
     */
    size_t EncoderWeightSize(arma::mat const &params) const
    {
        return (params.n_rows-1)/2 * (params.n_cols - 1);
    }

    /**
     * Get the number of elements of b1 of the parameters
     * @param params the parameters
     * @return element size of the bias
     */
    size_t EncoderBiasSize(arma::mat const &params) const
    {
        return (params.n_rows - 1) / 2;
    }

    /**
     * Get the number of elements of w1 and b1 of the parameters
     * @param params the parameters
     * @return element size of the bias and weights
     */
    size_t EncoderSize(arma::mat const &params) const
    {
        return (params.n_rows-1)/2 * (params.n_cols);
    }

    std::vector<arma::mat*> &trainData;
    arma::mat initialPoint;
    arma::mat softmaxParameters;
    std::vector<arma::mat*> paramArray;
    OutputLayerType &outLayerType;
    size_t LayerTypesParamSize;
    FineTuneGradient fineTuneGrad;
};

} // namespace nn
} // namespace mlpack

#include "finetune_impl.hpp"
