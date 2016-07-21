/**
 * @file softmax_regression_main.cpp
 * @author Tham Ngap Wei
 *
 * Main executable for softmax regression.
 *
 * This file is part of mlpack 2.0.3.
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
PARAM_STRING("training_file", "A file containing the training set (the matrix "
    "of predictors, X).", "t", "");
PARAM_STRING("labels_file", "A file containing labels (0 or 1) for the points "
    "in the training set (y). The labels must order as a row", "l", "");

// Model loading/saving.
PARAM_STRING("input_model_file", "File containing existing model (parameters).",
    "m", "");
PARAM_STRING("output_model_file", "File to save trained softmax regression "
    "model to.", "M", "");

// Testing.
PARAM_STRING("test_data", "File containing test dataset.", "T", "");
PARAM_STRING("predictions_file", "File to save predictions for test dataset "
    "into.", "p", "");
PARAM_STRING("test_labels", "File containing test labels.", "L", "");

// Softmax configuration options.
PARAM_INT("max_iterations", "Maximum number of iterations before termination.",
    "n", 400);

PARAM_INT("number_of_classes", "Number of classes for classification; if "
    "unspecified (or 0), the number of classes found in the labels will be "
    "used.", "c", 0);

PARAM_DOUBLE("lambda", "L2-regularization constant", "r", 0.0001);

PARAM_FLAG("no_intercept", "Do not add the intercept term to the model.", "N");

using namespace std;

// Count the number of classes in the given labels (if numClasses == 0).
size_t CalculateNumberOfClasses(const size_t numClasses,
                                const arma::Row<size_t>& trainLabels);

// Test the accuracy of the model.
template<typename Model>
void TestPredictAcc(const string& testFile,
                    const string& predictionsFile,
                    const string& testLabels,
                    const size_t numClasses,
                    const Model& model);

// Build the softmax model given the parameters.
template<typename Model>
unique_ptr<Model> TrainSoftmax(const string& trainingFile,
                               const string& labelsFile,
                               const string& inputModelFile,
                               const size_t maxIterations);

int main(int argc, char** argv)
{
  using namespace mlpack;

  CLI::ParseCommandLine(argc, argv);

  const string trainingFile = CLI::GetParam<string>("training_file");
  const string labelsFile = CLI::GetParam<string>("labels_file");
  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const string testLabelsFile = CLI::GetParam<string>("test_labels");
  const int maxIterations = CLI::GetParam<int>("max_iterations");
  const string predictionsFile = CLI::GetParam<string>("predictions_file");

  // One of inputFile and modelFile must be specified.
  if (!CLI::HasParam("input_model_file") && !CLI::HasParam("training_file"))
    Log::Fatal << "One of --input_model_file or --training_file must be specified."
        << endl;

  if ((CLI::HasParam("training_file") || CLI::HasParam("labels_file")) &&
      !(CLI::HasParam("training_file") && CLI::HasParam("labels_file")))
    Log::Fatal << "--labels_file must be specified with --training_file!"
        << endl;

  if (maxIterations < 0)
    Log::Fatal << "Invalid value for maximum iterations (" << maxIterations
        << ")! Must be greater than or equal to 0." << endl;

  // Make sure we have an output file of some sort.
  if (!CLI::HasParam("output_model_file") &&
      !CLI::HasParam("test_labels") &&
      !CLI::HasParam("predictions_file"))
    Log::Warn << "None of --output_model_file, --test_labels, or "
        << "--predictions_file are set; no results from this program will be "
        << "saved." << endl;


  using SM = regression::SoftmaxRegression<>;
  unique_ptr<SM> sm = TrainSoftmax<SM>(trainingFile,
                                            labelsFile,
                                            inputModelFile,
                                            maxIterations);

  TestPredictAcc(CLI::GetParam<string>("test_data"),
                 CLI::GetParam<string>("predictions_file"),
                 CLI::GetParam<string>("test_labels"),
                 sm->NumClasses(), *sm);

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
void TestPredictAcc(const string& testFile,
                    const string& predictionsFile,
                    const string& testLabelsFile,
                    size_t numClasses,
                    const Model& model)
{
  using namespace mlpack;

  // If there is no test set, there is nothing to test on.
  if (testFile.empty() && predictionsFile.empty() && testLabelsFile.empty())
    return;

  if (!testLabelsFile.empty() && testFile.empty())
  {
    Log::Warn << "--test_labels specified, but --test_file is not specified."
        << "  The parameter will be ignored." << endl;
    return;
  }

  if (!predictionsFile.empty() && testFile.empty())
  {
    Log::Warn << "--predictions_file specified, but --test_file is not "
        << "specified.  The parameter will be ignored." << endl;
    return;
  }

  // Get the test dataset, and get predictions.
  arma::mat testData;
  data::Load(testFile, testData, true);

  arma::Row<size_t> predictLabels;
  model.Predict(testData, predictLabels);

  // Save predictions, if desired.
  if (!predictionsFile.empty())
    data::Save(predictionsFile, predictLabels);

  // Calculate accuracy, if desired.
  if (!testLabelsFile.empty())
  {
    arma::Mat<size_t> tmpTestLabels;
    arma::Row<size_t> testLabels;
    data::Load(testLabelsFile, tmpTestLabels, true);
    testLabels = tmpTestLabels.row(0);

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
unique_ptr<Model> TrainSoftmax(const string& trainingFile,
                               const string& labelsFile,
                               const string& inputModelFile,
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
    arma::mat trainData;
    arma::Row<size_t> trainLabels;
    arma::Mat<size_t> tmpTrainLabels;

    //load functions of mlpack do not works on windows, it will complain
    //"[FATAL] Unable to detect type of 'softmax_data.txt'; incorrect extension?"
    data::Load(trainingFile, trainData, true);
    data::Load(labelsFile, tmpTrainLabels, true);
    trainLabels = tmpTrainLabels.row(0);

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
