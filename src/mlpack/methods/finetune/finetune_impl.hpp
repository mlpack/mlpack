namespace mlpack{
namespace nn{

template<typename OutputLayerType,
         typename OutputGradientFunction>
FineTuneFunction<OutputLayerType, OutputGradientFunction>::
FineTuneFunction(std::vector<arma::mat*> &input,
                 std::vector<arma::mat*> &parameters,
                 OutputLayerType &outLayerType)
    : trainData(input),
      paramArray(parameters),
      outLayerType(outLayerType),
      LayerTypesParamSize(TotalEncoderSize())
{
    InitializeWeights();
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
double FineTuneFunction<OutputLayerType, OutputGradientFunction>::
Evaluate(const arma::mat& parameters)
{
    UpdateInputData(parameters);
    std::copy(std::begin(parameters) + LayerTypesParamSize,
              std::end(parameters), std::begin(softmaxParameters));

    return outLayerType.Evaluate(softmaxParameters);
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
void FineTuneFunction<OutputLayerType, OutputGradientFunction>::
InitializeWeights()
{    
    arma::mat const &initialMatrix = *paramArray[paramArray.size() - 1];
    initialPoint.set_size(LayerTypesParamSize +
                          initialMatrix.n_elem, 1);
    softmaxParameters.set_size(arma::size(initialMatrix));
    FlattenLayerParams();

    std::copy(std::begin(initialMatrix), std::end(initialMatrix),
              std::begin(initialPoint) + LayerTypesParamSize);
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
void FineTuneFunction<OutputLayerType, OutputGradientFunction>::
Gradient(const arma::mat& parameters, arma::mat& gradient)
{        
    gradient.set_size(parameters.n_rows, parameters.n_cols);
    arma::mat derivative;
    outLayerType.Gradient(softmaxParameters, derivative);
    std::copy(std::begin(derivative), std::end(derivative),
              std::begin(gradient) + LayerTypesParamSize);

    arma::mat backPropGradient;
    fineTuneGrad.LastGradient(*trainData.back(), softmaxParameters,
                              outLayerType,
                              backPropGradient);

    size_t start = LayerTypesParamSize;
    for(int index = static_cast<int>(trainData.size() - 2);
        index >= 0; --index)
    {
        auto const &input = *trainData[index];
        arma::mat const b1 = backPropGradient * arma::ones(input.n_cols, 1);
        arma::mat const w1 = backPropGradient * input.t();
        std::copy(std::begin(b1),
                  std::end(b1),
                  std::begin(gradient) + start - b1.n_elem);
        std::copy(std::begin(w1),
                  std::end(w1),
                  std::begin(gradient) + start - b1.n_elem - w1.n_elem);
        if(index != 0){
            arma::mat const oldW1 =
                    parameters.submat(start - w1.n_elem - b1.n_elem, 0,
                                      start - b1.n_elem - 1, 0);
            size_t const hiddenSize = HiddenSize(*paramArray[index]);
            size_t const visibleSize = VisibleSize(*paramArray[index]);
            fineTuneGrad.Gradient(input,
                                  arma::reshape(oldW1, hiddenSize, visibleSize),
                                  backPropGradient,
                                  backPropGradient);
        }
        start = start - w1.n_elem - b1.n_elem;
    }
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
size_t FineTuneFunction<OutputLayerType, OutputGradientFunction>::
TotalEncoderSize() const
{
    size_t totalSize = 0;
    for(size_t i = 0; i != paramArray.size() - 1; ++i){
        totalSize += EncoderSize(*paramArray[i]);
    }
    return totalSize;
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
void FineTuneFunction<OutputLayerType, OutputGradientFunction>::
UpdateInputData(arma::mat const &parameters)
{
    size_t start = 0;
    for(size_t i = 0; i != paramArray.size() - 1; ++i)
    {
        const size_t w1End = start + EncoderWeightSize(*paramArray[i]);
        const size_t b1End = w1End + EncoderBiasSize(*paramArray[i]);
        const size_t hiddenSize = HiddenSize(*paramArray[i]);
        const size_t visibleSize = VisibleSize(*paramArray[i]);
        auto const &w1 = parameters.submat(start, 0, w1End - 1, 0);
        auto const &b1 = parameters.submat(w1End, 0, b1End - 1, 0);
        Sigmoid(arma::reshape(w1, hiddenSize, visibleSize) * *trainData[i] +
                arma::repmat(b1, 1, (*trainData[i]).n_cols),
                *trainData[i + 1]);
        start = b1End;
    }
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
void FineTuneFunction<OutputLayerType, OutputGradientFunction>::
FlattenLayerParams()
{
    // The sparse autoencoder module uses a matrix to store
    // the parameters, its structure looks like:
    //          vSize   1
    //       |        |  |
    //  hSize|   w1   |b1|
    //       |________|__|
    //       |        |  |
    //  hSize|   w2'  |  |
    //       |________|__|
    //      1|   b2'  |  |
    size_t start = 0;
    for(size_t i = 0; i != paramArray.size() - 1; ++i)
    {
        arma::mat const &params = *paramArray[i];
        //copy w1 to initialPoint
        auto const &w1 = params.submat(0, 0,
                                       (params.n_rows - 1)/2 - 1,
                                       params.n_cols - 2);

        initialPoint.submat(start, 0,
                            start + EncoderWeightSize(params) - 1, 0) =
                arma::reshape(w1,
                              EncoderWeightSize(params), 1);

        //copy b1 to initialPoint
        start += EncoderWeightSize(params);
        auto const &b1 = params.submat(0, params.n_cols - 1,
                                       (params.n_rows - 1)/2 - 1,
                                       params.n_cols - 1);
        initialPoint.submat(start, 0,
                            start + EncoderBiasSize(params) - 1, 0) = b1;

        start += b1.n_elem;
    }
}

template<typename OutputLayerType,
         typename OutputGradientFunction>
void FineTuneFunction<OutputLayerType, OutputGradientFunction>::
UpdateParameters(arma::mat const &parameters)
{
    size_t start = 0;
    for(size_t i = 0; i != paramArray.size() - 1; ++i)
    {
        arma::mat &param = *paramArray[i];
        size_t const end = start + EncoderSize(param) - 1;
        param.submat(0, 0, HiddenSize(param) - 1,
                     VisibleSize(param)) =
                arma::reshape(parameters.submat(start, 0, end, 0),
                              HiddenSize(param), VisibleSize(param) + 1);
        start += EncoderSize(param);
    }
    std::copy(std::begin(parameters) + LayerTypesParamSize,
              std::end(parameters), std::begin(*paramArray.back()));
}

} // namespace nn
} // namespace mlpack
