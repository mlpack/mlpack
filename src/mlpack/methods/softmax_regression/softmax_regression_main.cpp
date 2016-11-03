/**
 * @file softmax_regression_main.cpp
 *
 * Main program for softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include <memory>
#include <set>

// Define parameters for the executable.
PROGRAM_INFO("Softmax Regression", "This program performs softmax regression, "
    "a generalization of logistic regression to the multiclass case, and has "
    "support for L2 regularization.  The program is able to train a model, load"
    " an existing model, and give predictions (and optionally their accuracy) "
    "for test data."
    "\n\n"
    "Training a softmax regression model is done by giving a file of training "
    "points with --training_file (-t) and their corresponding labels with "
    "--labels_file (-l).  The number of classes can be manually specified with "
    "the --number_of_classes (-n) option, and the maximum number of iterations "
    "of the L-BFGS optimizer can be specified with the --max_iterations (-M) "
    "option.  The L2 regularization constant can be specified with --lambda "
    "(-r), and if an intercept term is not desired in the model, the "
    "--no_intercept (-N) can be specified."
    "\n\n"
    "The trained model can be saved to a file with the --output_model_file (-m) "
    "option.  If training is not desired, but only testing is, a model can be "
    "loaded with the --input_model_file (-i) option.  At the current time, a loaded "
    "model cannot be trained further, so specifying both -i and -t is not "
    "allowed."
    "\n\n"
    "The program is also able to evaluate a model on test data.  A test dataset"
    " can be specified with the --test_data (-T) option.  Class predictions "
    "will be saved in the file specified with the --predictions_file (-p) "
    "option.  If labels are specified for the test data, with the --test_labels"
    " (-L) option, then the program will print the accuracy of the predictions "
    "on the given test set and its corresponding labels.");

// Required options.
PARAM_MATRIX_IN("training", "A matrix containing the training set (the matrix "
    "of predictors, X).", "t");
PARAM_UMATRIX_IN("labels", "A matrix containing labels (0 or 1) for the points "
    "in the training set (y). The labels must order as a row.", "l");

// Model loading/saving.
PARAM_STRING_IN("input_model_file", "File containing existing model "
    "(parameters).", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained softmax regression "
    "model to.", "M");

// Testing.
PARAM_MATRIX_IN("test", "Matrix containing test dataset.", "T");
PARAM_UMATRIX_OUT("predictions", "Matrix to save predictions for test dataset "
    "into.", "p");
PARAM_UMATRIX_IN("test_labels", "Matrix containing test labels.", "L");

// Softmax configuration options.
PARAM_INT_IN("max_iterations", "Maximum number of iterations before "
    "termination.", "n", 400);

PARAM_INT_IN("number_of_classes", "Number of classes for classification; if "
    "unspecified (or 0), the number of classes found in the labels will be "
    "used.", "c", 0);

PARAM_DOUBLE_IN("lambda", "L2-regularization constant", "r", 0.0001);

PARAM_FLAG("no_intercept", "Do not add the intercept term to the model.", "N");

using namespace std;

// Count the number of classes in the given labels (if numClasses == 0).
size_t CalculateNumberOfClasses(const size_t numClasses,
                                const arma::Row<size_t>& trainLabels);

// Test the accuracy of the model.
template<typename Model>
void TestPredictAcc(const size_t numClasses, const Model& model);

// Build the softmax model given the parameters.
template<typename Model>
unique_ptr<Model> TrainSoftmax(const string& inputModelFile,
                               const size_t maxIterations);

int main(int argc, char** argv)
{
  using namespace mlpack;

  CLI::ParseCommandLine(argc, argv);

  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const int maxIterations = CLI::GetParam<int>("max_iterations");

  // One of inputFile and modelFile must be specified.
  if (!CLI::HasParam("input_model_file") && !CLI::HasParam("training"))
    Log::Fatal << "One of --input_model_file or --training_file must be "
        << "specified." << endl;

  if ((CLI::HasParam("training") || CLI::HasParam("labels")) &&
      !(CLI::HasParam("training") && CLI::HasParam("labels")))
    Log::Fatal << "--labels_file must be specified with --training_file!"
        << endl;

  if (maxIterations < 0)
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations
        << ")! Must be greater than or equal to 0." << endl;

  // Make sure we have an output file of some sort.
  if (!CLI::HasParam("output_model_file") &&
      !CLI::HasParam("predictions"))
    Log::Warn << "Neither --output_model_file nor --predictions_file are set; "
        << "no results from this program will be saved." << endl;

  using SM = regression::SoftmaxRegression<>;
  unique_ptr<SM> sm = TrainSoftmax<SM>(inputModelFile, maxIterations);

  TestPredictAcc(sm->NumClasses(), *sm);

  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"),
        "softmax_regression_model", *sm, true);
}

size_t CalculateNumberOfClasses(const size_t numClasses,
                                const arma::Row<size_t>& trainLabels)
{
  if (numClasses == 0)
  {
    const set<size_t> unique_labels(begin(trainLabels),
                                    end(trainLabels));
    return unique_labels.size();
  }
  else
  {
    return numClasses;
  }
}

template<typename Model>
void TestPredictAcc(size_t numClasses, const Model& model)
{
  using namespace mlpack;

  // If there is no test set, there is nothing to test on.
  if (!CLI::HasParam("test") && !CLI::HasParam("predictions") &&
      !CLI::HasParam("test_labels"))
    return;

  if (CLI::HasParam("test_labels") && !CLI::HasParam("test"))
  {
    Log::Warn << "--test_labels specified, but --test_file is not specified."
        << "  The parameter will be ignored." << endl;
    return;
  }

  if (CLI::HasParam("predictions") && !CLI::HasParam("test"))
  {
    Log::Warn << "--predictions_file specified, but --test_file is not "
        << "specified.  The parameter will be ignored." << endl;
    return;
  }

  // Get the test dataset, and get predictions.
  arma::mat testData = std::move(CLI::GetParam<arma::mat>("test"));

  arma::Row<size_t> predictLabels;
  model.Predict(testData, predictLabels);

  // Save predictions, if desired.
  if (CLI::HasParam("predictions"))
    CLI::GetParam<arma::Mat<size_t>>("predictions") = std::move(predictLabels);

  // Calculate accuracy, if desired.
  if (CLI::HasParam("test_labels"))
  {
    arma::Mat<size_t> tmpTestLabels =
        CLI::GetParam<arma::Mat<size_t>>("test_labels");
    arma::Row<size_t> testLabels = tmpTestLabels.row(0);

    if (testData.n_cols != testLabels.n_elem)
    {
      Log::Fatal << "Test data in --test_data has " << testData.n_cols
          << " points, but labels in --test_labels have "
          << testLabels.n_elem << " labels!" << endl;
    }

    vector<size_t> bingoLabels(numClasses, 0);
    vector<size_t> labelSize(numClasses, 0);
    for (arma::uword i = 0; i != predictLabels.n_elem; ++i)
    {
      if (predictLabels(i) == testLabels(i))
      {
        ++bingoLabels[testLabels(i)];
      }
      ++labelSize[testLabels(i)];
    }

    size_t totalBingo = 0;
    for (size_t i = 0; i != bingoLabels.size(); ++i)
    {
      Log::Info << "Accuracy for points with label " << i << " is "
          << (bingoLabels[i] / static_cast<double>(labelSize[i])) << " ("
          << bingoLabels[i] << " of " << labelSize[i] << ")." << endl;
      totalBingo += bingoLabels[i];
    }

    Log::Info << "Total accuracy for all points is "
        << (totalBingo) / static_cast<double>(predictLabels.n_elem) << " ("
        << totalBingo << " of " << predictLabels.n_elem << ")." << endl;
  }
}

template<typename Model>
unique_ptr<Model> TrainSoftmax(const std::string& inputModelFile,
                               const size_t maxIterations)
{
  using namespace mlpack;

  using SRF = regression::SoftmaxRegressionFunction;

  unique_ptr<Model> sm;
  if (!inputModelFile.empty())
  {
    sm.reset(new Model(0, 0, false));
    mlpack::data::Load(inputModelFile, "softmax_regression_model", *sm, true);
  }
  else
  {
    arma::mat trainData = std::move(CLI::GetParam<arma::mat>("training"));
    arma::Mat<size_t> tmpTrainLabels =
        std::move(CLI::GetParam<arma::Mat<size_t>>("labels"));
    arma::Row<size_t> trainLabels = tmpTrainLabels.row(0);

    if (trainData.n_cols != trainLabels.n_elem)
      Log::Fatal << "Samples of input_data should same as the size of "
          << "input_label." << endl;

    const size_t numClasses = CalculateNumberOfClasses(
        (size_t) CLI::GetParam<int>("number_of_classes"), trainLabels);

    const bool intercept = CLI::HasParam("no_intercept") ? false : true;

    SRF smFunction(trainData, trainLabels, numClasses, intercept,
        CLI::GetParam<double>("lambda"));

    const size_t numBasis = 5;
    optimization::L_BFGS<SRF> optimizer(smFunction, numBasis, maxIterations);
    sm.reset(new Model(optimizer));
  }

  return sm;
}
