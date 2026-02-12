#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME ann

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/init_rules.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/negative_log_likelihood.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/layer/elu.hpp>
#include <mlpack/methods/ann/layer/hard_tanh.hpp>
#include <mlpack/methods/ann/layer/relu6.hpp>
#include <mlpack/methods/ann/layer/parametric_relu.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>
#include <mlpack/methods/ann/layer/log_softmax.hpp>
#include <mlpack/methods/ann/layer/dropout.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/identity.hpp>


using namespace mlpack;
using namespace std;

// Model type for serialization.
// We use MeanSquaredError to support both regression and classification.
using FFNModel = FFN<MeanSquaredError, RandomInitialization>;

// Program Name.
BINDING_USER_NAME("Artificial Neural Network");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of feed-forward neural networks for classification and "
    "regression tasks.");

// Long description.
BINDING_LONG_DESC(
    "This program trains and uses feed-forward neural networks.");

// Example.
BINDING_EXAMPLE(
    "Train a model:\n"
    "$ mlpack_ann -t data.csv -l labels.csv -L layers.txt -i 4 -O 3 -M "
    "model.bin");

PARAM_MATRIX_IN("training", "Matrix of training data.", "t");
PARAM_MATRIX_IN("labels", "Labels of training data.", "l");
PARAM_MATRIX_IN("test", "Matrix of test data.", "T");
PARAM_MODEL_IN(FFNModel, "input_model", "Pre-trained model to use.", "m");
PARAM_MODEL_OUT(FFNModel, "output_model", "output trained model", "M");
PARAM_MATRIX_OUT("predictions", "predictions on test set.", "p");
PARAM_STRING_IN("optimizer", "Optimizer to use for training.", "o", "rmsprop");
PARAM_DOUBLE_IN("step_size", "Step size for optimizer.", "s", 0.01);
PARAM_INT_IN("max_iterations", "Maximum iterations for training.", "n", 1000);
PARAM_INT_IN("batch_size", "Batch size for training.", "b", 32);
PARAM_STRING_IN("layers_file", "file defining network architecture.", "L", "");
PARAM_INT_IN("input_dim", "Input dimension (required if creating new model).",
             "i", 0);
PARAM_INT_IN("output_dim", "Output dimension (required if creating new model).",
             "O", 0);
PARAM_FLAG("regression", "Perform regression (default is classification).", "r");
PARAM_INT_IN("seed", "Random seed.", "S", 0);

// function to make layer specifications
template<typename MatType>
void AddLayerFromString(FFNModel& model, const string& layerSpec)
{
  istringstream iss(layerSpec);
  string layerType;
  iss >> layerType;

  // Adding dense layer
  if (layerType == "Linear")
  {
    size_t out;
    iss >> out;
    model.Add<Linear<>>(out);
  }
  // Identity
  else if (layerType == "Identity")
  {
    model.Add<Identity<>>();
  }
  // Activations
  else if (layerType == "ReLU")
  {
    model.Add<ReLU<>>();
  }
  else if (layerType == "LeakyReLU")
  {
    double alpha = 0.01;
    iss >> alpha;
    model.Add<LeakyReLU<>>(alpha);
  }
  else if (layerType == "ELU")
  {
    double alpha = 1.0;
    iss >> alpha;
    model.Add<ELU<>>(alpha);
  }
  else if (layerType == "Sigmoid")
  {
    model.Add<Sigmoid<>>();
  }
  else if (layerType == "Tanh")
  {
    model.Add<TanH<>>();
  }
  else if (layerType == "HardTanH")
  {
    model.Add<HardTanH<>>();
  }
  else if (layerType == "ReLU6")
  {
    model.Add<ReLU6<>>();
  }
  else if (layerType == "PReLU")
  {
    double alpha = 0.25;
    iss >> alpha;
    model.Add<PReLU<>>(alpha);
  }
  // output activations
  else if (layerType == "Softmax")
  {
    model.Add<Softmax<>>();
  }
  else if (layerType == "LogSoftmax")
  {
    model.Add<LogSoftMax<>>();
  }
  // regularizations
  else if (layerType == "Dropout")
  {
    double ratio = 0.5;
    iss >> ratio;
    model.Add<Dropout<>>(ratio);
  }
  else if (layerType == "BatchNorm")
  {
    model.Add<BatchNorm<>>();
  }
  // Error
  else
  {
    std::cerr << "Log::Fatal at AddLayerFromString: Unknown layer type: " << layerType << std::endl;
    Log::Fatal << "Unknown layer type: " << layerType << endl;
  }
}

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));

  // Check parameters
  const bool training = params.Has("training");
  const bool testing = params.Has("test");
  const bool loadModel = params.Has("input_model");

  RequireAtLeastOnePassed(params, { "training", "test" }, true,
      "must specify either training or test data");

  if (training)
  {
    RequireAtLeastOnePassed(params, { "labels" }, true,
        "labels must be specified for training");
  }

  // Variables for training data (loaded up front to allow dim inference)
  arma::mat trainData;
  arma::mat trainLabels;

  // Load data if training
  if (training)
  {
    trainData = std::move(params.Get<arma::mat>("training"));
    const arma::mat labels = std::move(params.Get<arma::mat>("labels"));

    // Process labels
    if (!params.Has("regression"))
    {
      // Convert to one-hot encoding for classification with MSE.
      // Assume labels are class indices 0..K-1 or 1..K in first row.
      // (mlpack usually uses 0-based for internal logic but 1-based labels are common)
      // Here we trust the user to provide 0-based classes or we shift if min >= 1?
      // For safety, assume 0-based.
      
      arma::Row<size_t> labelIndices = 
          arma::conv_to<arma::Row<size_t>>::from(labels.row(0));
      
      size_t numClasses = 0;
      if (labelIndices.n_elem > 0)
        numClasses = arma::max(labelIndices) + 1;
      
      if (params.Has("output_dim"))
      {
         const size_t dim = (size_t)params.Get<int>("output_dim");
         if (dim > numClasses) numClasses = dim;
      }

      trainLabels.zeros(numClasses, labels.n_cols);
      for(size_t i = 0; i < labels.n_cols; ++i)
      {
        if (labelIndices[i] < numClasses)
          trainLabels(labelIndices[i], i) = 1;
        else {
          Log::Fatal << "Label index " << labelIndices[i] << " out of bounds (0.." 
                     << numClasses - 1 << ")." << endl;
        }
      }
    }
    else
    {
      // Regression: Use labels directly
      trainLabels = labels;
    }
  }

  // Load or create model
  FFNModel model;

  if (loadModel)
  {
    model = std::move(params.Get<FFNModel>("input_model"));
    Log::Info << "Loaded model from file." << endl;
  }
  else if (training)
  {
    // Creating new model
    RequireAtLeastOnePassed(params, { "layers_file" }, true,
        "layers_file must be specified when creating a new model");

    // Infer or get dimensions
    size_t inputDim = 0;
    if (params.Has("input_dim"))
    {
      inputDim = (size_t) params.Get<int>("input_dim");
    }
    else
    {
      inputDim = trainData.n_rows;
      Log::Info << "Inferred input dimension from data: " << inputDim << endl;
    }

    size_t outputDim = 0;
    if (params.Has("output_dim"))
    {
      outputDim = (size_t) params.Get<int>("output_dim");
    }
    else
    {
      outputDim = trainLabels.n_rows;
      Log::Info << "Inferred output dimension from labels: " << outputDim
          << endl;
    }

    model.InputDimensions() = { inputDim };

    // Parse layers file
    const string layersFile = params.Get<string>("layers_file");
    ifstream ifs(layersFile);

    if (!ifs.is_open()) {
      Log::Fatal << "Cannot open layers file: " << layersFile << endl;
    }

    string line;
    while (getline(ifs, line))
    {
      if (!line.empty() && line[0] != '#') {
        AddLayerFromString<arma::mat>(model, line);
      }
    }

    Log::Info << "Created new model with " << model.Network().size()
        << " layers." << endl;
  }

  // Training
  if (training)
  {
    const string optimizerType = params.Get<string>("optimizer");
    const double stepSize = params.Get<double>("step_size");
    const size_t maxIterations = params.Get<int>("max_iterations");
    const size_t batchSize = params.Get<int>("batch_size");

    Log::Info << "Training model..." << endl;

    if (optimizerType == "sgd")
    {
      ens::SGD optimizer(stepSize, batchSize, maxIterations);
      model.Train(trainData, trainLabels, optimizer);
    }
    else if (optimizerType == "adam")
    {
      ens::Adam optimizer(stepSize, batchSize, 0.9, 0.999, 1e-8,
          maxIterations);
      model.Train(trainData, trainLabels, optimizer);
    }
    else if (optimizerType == "rmsprop")
    {
      ens::RMSProp optimizer(stepSize, batchSize, 0.99, 1e-8, maxIterations);
      model.Train(trainData, trainLabels, optimizer);
    }
    else
    {
      Log::Fatal << "Unknown optimizer: " << optimizerType << endl;
    }

    Log::Info << "Training complete." << endl;
  }

  // Testing/Prediction
  if (testing)
  {
    arma::mat testData = std::move(params.Get<arma::mat>("test"));
    arma::mat predictions;

    Log::Info << "Making predictions..." << endl;
    model.Predict(testData, predictions);

    // For classification, get class labels
    if (!params.Has("regression"))
    {
      arma::Row<size_t> classLabels;
      // If LogSoftmax output, exp it. If Linear output, just use it.
      // Usually index_max works on logic-like output too.
      // We assume output is probabilities or scores for each class.
      arma::urowvec indices = arma::index_max(predictions, 0);
      params.Get<arma::mat>("predictions") =
          arma::conv_to<arma::mat>::from(indices);
    }
    else
    {
      params.Get<arma::mat>("predictions") = std::move(predictions);
    }

    Log::Info << "Predictions complete." << endl;
  }


  // Save model
  if (params.Has("output_model"))
  {
    try
    {
      params.Get<FFNModel>("output_model") = std::move(model);
      Log::Info << "Model saved." << endl;
    }
    catch (std::exception& e)
    {
      Log::Warn << "Failed to save model: " << e.what() << endl;
      Log::Warn << "This is expected if MLPACK_ENABLE_ANN_SERIALIZATION is "
          << "not defined." << endl;
    }
  }
}
