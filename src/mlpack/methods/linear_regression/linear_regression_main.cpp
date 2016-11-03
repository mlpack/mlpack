/**
 * @file linear_regression_main.cpp
 * @author James Cline
 *
 * Main function for least-squares linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "linear_regression.hpp"

PROGRAM_INFO("Simple Linear Regression and Prediction",
    "An implementation of simple linear regression and simple ridge regression "
    "using ordinary least squares. This solves the problem\n\n"
    "  y = X * b + e\n\n"
    "where X (--input_file) and y (the last column of --input_file, or "
    "--input_responses) are known and b is the desired variable.  If the "
    "covariance matrix (X'X) is not invertible, or if the solution is "
    "overdetermined, then specify a Tikhonov regularization constant (--lambda)"
    " greater than 0, which will regularize the covariance matrix to make it "
    "invertible.  The calculated b is saved to disk (--output_file).\n"
    "\n"
    "Optionally, the calculated value of b is used to predict the responses for"
    " another matrix X' (--test_file):\n\n"
    "   y' = X' * b\n\n"
    "and these predicted responses, y', are saved to a file "
    "(--output_predictions).  This type of regression is related to "
    "least-angle regression, which mlpack implements with the 'lars' "
    "executable.");

PARAM_MATRIX_IN("training", "Matrix containing training set X (regressors).",
    "t");
PARAM_MATRIX_IN("training_responses", "Optional matrix containing y "
    "(responses). If not given, the responses are assumed to be the last row "
    "of the input file.", "r");

PARAM_STRING_IN("input_model_file", "File containing existing model "
    "(parameters).", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained model to.", "M");

PARAM_MATRIX_IN("test", "Matrix containing X' (test regressors).", "T");

// This is the future name of the parameter.
PARAM_TMATRIX_OUT("output_predictions", "If --test_file is specified, this "
    "matrix is where the predicted responses will be saved.", "o");

PARAM_DOUBLE_IN("lambda", "Tikhonov regularization for ridge regression.  If 0,"
    " the method reduces to linear regression.", "l", 0.0);

using namespace mlpack;
using namespace mlpack::regression;
using namespace arma;
using namespace std;

int main(int argc, char* argv[])
{
  // Handle parameters.
  CLI::ParseCommandLine(argc, argv);

  const string inputModelFile = CLI::GetParam<string>("input_model_file");
  const string outputModelFile = CLI::GetParam<string>("output_model_file");
  const double lambda = CLI::GetParam<double>("lambda");

  if (!CLI::HasParam("test") && CLI::HasParam("output_predictions"))
    Log::Warn << "--output_predictions_file (-o) ignored because --test_file "
        << "(-T) is not specified." << endl;

  mat regressors;
  mat responses;

  LinearRegression lr;
  lr.Lambda() = lambda;

  bool computeModel = false;

  // We want to determine if an input file XOR model file were given.
  if (!CLI::HasParam("training"))
  {
    if (!CLI::HasParam("input_model_file"))
      Log::Fatal << "You must specify either --input_file or --model_file."
          << endl;
    else // The model file was specified, no problems.
      computeModel = false;
  }
  // The user specified an input file but no model file, no problems.
  else if (!CLI::HasParam("input_model_file"))
    computeModel = true;
  // The user specified both an input file and model file.
  // This is ambiguous -- which model should we use? A generated one or given
  // one?  Report error and exit.
  else
  {
    Log::Fatal << "You must specify either --input_file or --model_file, not "
        << "both." << endl;
  }

  if (CLI::HasParam("test") && !CLI::HasParam("output_predictions"))
    Log::Warn << "--test_file (-t) specified, but --output_predictions_file "
        << "(-o) is not; no results will be saved." << endl;

  // If they specified a model file, we also need a test file or we
  // have nothing to do.
  if (!computeModel && !CLI::HasParam("test"))
  {
    Log::Fatal << "When specifying --model_file, you must also specify "
        << "--test_file." << endl;
  }

  if (!computeModel && CLI::HasParam("lambda"))
  {
    Log::Warn << "--lambda ignored because no model is being trained." << endl;
  }

  if (!CLI::HasParam("output_model_file") &&
      !CLI::HasParam("output_predictions"))
  {
    Log::Warn << "Neither --output_model_file nor --output_predictions_file are"
        << " specified; no output will be saved!" << endl;
  }

  // An input file was given and we need to generate the model.
  if (computeModel)
  {
    Timer::Start("load_regressors");
    regressors = std::move(CLI::GetParam<mat>("training"));
    Timer::Stop("load_regressors");

    // Are the responses in a separate file?
    if (!CLI::HasParam("training_responses"))
    {
      // The initial predictors for y, Nx1.
      responses = trans(regressors.row(regressors.n_rows - 1));
      regressors.shed_row(regressors.n_rows - 1);
    }
    else
    {
      // The initial predictors for y, Nx1.
      Timer::Start("load_responses");
      responses = std::move(CLI::GetParam<mat>("training_responses"));
      Timer::Stop("load_responses");

      if (responses.n_rows == 1)
        responses = trans(responses); // Probably loaded backwards.

      if (responses.n_cols > 1)
        Log::Fatal << "The responses must have one column." << endl;

      if (responses.n_rows != regressors.n_cols)
        Log::Fatal << "The responses must have the same number of rows as the "
            "training file." << endl;
    }

    Timer::Start("regression");
    lr = LinearRegression(regressors, responses.unsafe_col(0));
    Timer::Stop("regression");

    // Save the parameters.
    if (CLI::HasParam("output_model_file"))
      data::Save(outputModelFile, "linearRegressionModel", lr);
  }

  // Did we want to predict, too?
  if (CLI::HasParam("test"))
  {
    // A model file was passed in, so load it.
    if (!computeModel)
    {
      Timer::Start("load_model");
      data::Load(inputModelFile, "linearRegressionModel", lr, true);
      Timer::Stop("load_model");
    }

    // Load the test file data.
    Timer::Start("load_test_points");
    mat points = std::move(CLI::GetParam<mat>("test"));
    Timer::Stop("load_test_points");

    // Ensure that test file data has the right number of features.
    if ((lr.Parameters().n_elem - 1) != points.n_rows)
    {
      Log::Fatal << "The model was trained on " << lr.Parameters().n_elem - 1
          << "-dimensional data, but the test points in '"
          << CLI::GetUnmappedParam<mat>("test") << "' are " << points.n_rows
          << "-dimensional!" << endl;
    }

    // Perform the predictions using our model.
    vec predictions;
    Timer::Start("prediction");
    lr.Predict(points, predictions);
    Timer::Stop("prediction");

    // Save predictions.
    if (CLI::HasParam("output_predictions"))
      CLI::GetParam<mat>("output_predictions") = std::move(predictions);
  }
}
