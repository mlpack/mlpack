/**
 * @file logistic_regression_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for logistic regression.
 *
 * This file is part of MLPACK 1.0.8.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>
#include "logistic_regression.hpp"

#include <mlpack/core/optimizers/sgd/sgd.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::optimization;

PROGRAM_INFO("L2-regularized Logistic Regression and Prediction",
    "An implementation of L2-regularized logistic regression using either the "
    "L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the "
    "regression problem\n"
    "\n"
    "  y = (1 / 1 + e^-(X * b))\n"
    "\n"
    "where y takes values 0 or 1.  Training the model is done by giving labeled"
    " data and iteratively training the parameters vector b.  The matrix of "
    "predictors (or features) X is specified with the --input_file option, and "
    "the vector of responses y is either the last column of the matrix given "
    "with --input_file, or a separate one-column vector given with the "
    "--input_responses option.  After training, the calculated b is saved to "
    "the file specified by --output_file.  An initial guess for b can be "
    "specified when the --model_file parameter is given with --input_file or "
    "--input_responses.  The tolerance of the optimizer can be set with "
    "--tolerance; the maximum number of iterations of the optimizer can be set "
    "with --max_iterations; and the type of the optimizer (SGD / L-BFGS) can be"
    " set with " "the --optimizer option.  Both the SGD and L-BFGS optimizers "
    "have more options, but the C++ interface must be used for those.  For the "
    "SGD optimizer, the --step_size parameter controls the step size taken at "
    "each iteration by the optimizer.  If the objective function for your data "
    "is oscillating between Inf and 0, the step size is probably too large.\n"
    "\n"
    "This implementation of logistic regression supports L2-regularization, "
    "which can help the parameter vector b from overfitting.  This parameter "
    "is specified with the --lambda option; by default, it is 0 (which means "
    "no regularization is performed).\n"
    "\n"
    "Optionally, the calculated value of b is used to predict the responses "
    "for another matrix of data points, if --test_file is specified.  The "
    "--test_file option can be specified without --input_file, so long as an "
    "existing logistic regression model is given with --model_file.  The "
    "output predictions from the logistic regression model are stored in the "
    "file given with --output_predictions.\n"
    "\n"
    "This implementation of logistic regression does not support the general "
    "multi-class case but instead only the two-class case.  Any responses must "
    "be either 0 or 1.");

PARAM_STRING("input_file", "File containing X (predictors).", "i", "");
PARAM_STRING("input_responses", "Optional file containing y (responses).  If "
    "not given, the responses are assumed to be the last column of the input "
    "file.", "r", "");

PARAM_STRING("model_file", "File containing existing model (parameters).", "m",
    "");

PARAM_STRING("output_file", "File where parameters (b) will be saved.", "o",
    "");

PARAM_STRING("test_file", "File containing test dataset.", "t", "");
PARAM_STRING("output_predictions", "If --test_file is specified, this file is "
    "where the predicted responses will be saved.", "p", "predictions.csv");
PARAM_DOUBLE("decision_boundary", "Decision boundary for prediction; if the "
    "logistic function for a point is less than the boundary, the class is "
    "taken to be 0; otherwise, the class is 1.", "d", 0.5);

PARAM_DOUBLE("lambda", "L2-regularization parameter for training.", "l", 0.0);
PARAM_STRING("optimizer", "Optimizer to use for training ('lbfgs' or 'sgd').",
    "O", "lbfgs");
PARAM_DOUBLE("tolerance", "Convergence tolerance for optimizer.", "T", 1e-10);
PARAM_INT("max_iterations", "Maximum iterations for optimizer (0 indicates no "
    "limit).", "M", 0);
PARAM_DOUBLE("step_size", "Step size for SGD optimizer.", "s", 0.01);

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Collect command-line options.
  const string inputFile = CLI::GetParam<string>("input_file");
  const string inputResponsesFile = CLI::GetParam<string>("input_responses");
  const string modelFile = CLI::GetParam<string>("model_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string testFile = CLI::GetParam<string>("test_file");
  const string outputPredictionsFile =
      CLI::GetParam<string>("output_predictions");
  const double lambda = CLI::GetParam<double>("lambda");
  const string optimizerType = CLI::GetParam<string>("optimizer");
  const double tolerance = CLI::GetParam<double>("tolerance");
  const size_t maxIterations = (size_t) CLI::GetParam<int>("max_iterations");
  const double decisionBoundary = CLI::GetParam<double>("decision_boundary");
  const double stepSize = CLI::GetParam<double>("step_size");

  // One of inputFile and modelFile must be specified.
  if (inputFile.empty() && modelFile.empty())
    Log::Fatal << "One of --model_file or --input_file must be specified."
        << endl;

  // If inputFile is specified, it must also have some responses with it.
  if (!inputFile.empty() && inputResponsesFile.empty())
    Log::Fatal << "If --input_file is specified, then --input_responses must "
        << "also be specified." << endl;

  // If they want predictions, they should supply a file to save them to.  This
  // is only a warning because the program can still work.
  if (!testFile.empty() && outputPredictionsFile.empty())
    Log::Warn << "--output_predictions not specified; predictions will not be "
        << "saved." << endl;

  // If no output file is given, the user should know that the model will not be
  // saved, but only if a model is being trained.
  if (outputFile.empty() && !inputFile.empty())
    Log::Warn << "--output_file not given; trained model will not be saved."
        << endl;

  // Tolerance needs to be positive.
  if (tolerance < 0.0)
    Log::Fatal << "Tolerance must be positive (received " << tolerance << ")."
        << endl;

  // Optimizer has to be L-BFGS or SGD.
  if (optimizerType != "lbfgs" && optimizerType != "sgd")
    Log::Fatal << "--optimizer must be 'lbfgs' or 'sgd'." << endl;

  // Lambda must be positive.
  if (lambda < 0.0)
    Log::Fatal << "L2-regularization parameter (--lambda) must be positive ("
        << "received " << lambda << ")." << endl;

  // Decision boundary must be between 0 and 1.
  if (decisionBoundary < 0.0 || decisionBoundary > 1.0)
    Log::Fatal << "Decision boundary (--decision_boundary) must be between 0.0 "
        << "and 1.0 (received " << decisionBoundary << ")." << endl;

  if ((stepSize < 0.0) && (optimizerType == "sgd"))
    Log::Fatal << "Step size (--step_size) must be positive (received "
        << stepSize << ")." << endl;

  // These are the matrices we might use.
  arma::mat regressors;
  arma::mat responses;
  arma::mat model;
  arma::mat testSet;
  arma::vec predictions;

  // Load matrices.
  if (!inputFile.empty())
    data::Load(inputFile, regressors, true);
  if (!inputResponsesFile.empty())
  {
    data::Load(inputResponsesFile, responses, true);
    if (responses.n_rows == 1)
      responses = responses.t();
    if (responses.n_rows != regressors.n_cols)
      Log::Fatal << "The responses (--input_responses) must have the same "
          << "number of points as the input dataset (--input_file)." << endl;
  }
  if (!testFile.empty())
    data::Load(testFile, testSet, true);
  if (!modelFile.empty())
  {
    data::Load(modelFile, model, true);
    if (model.n_rows == 1)
      model = model.t();
    if ((!regressors.empty()) && (model.n_rows != regressors.n_rows + 1))
      Log::Fatal << "The model (--model) must have dimensionality of one more "
          << "than the input dataset (the extra dimension is the intercept)."
          << endl;
    if ((!testSet.empty()) && (model.n_rows != testSet.n_rows + 1))
      Log::Fatal << "The model (--model) must have dimensionality of one more "
          << "than the test dataset (the extra dimension is the intercept)."
          << endl;
  }

  if (!regressors.empty())
  {
    // We need to train the model.  Prepare the optimizers.
    arma::vec responsesVec = responses.unsafe_col(0);
    LogisticRegressionFunction lrf(regressors, responsesVec, lambda);
    // Set the initial point, if necessary.
    if (!model.empty())
    {
      lrf.InitialPoint() = model;
      Log::Info << "Using model from '" << modelFile << "' as initial model "
          << "for training." << endl;
    }

    if (optimizerType == "lbfgs")
    {
      L_BFGS<LogisticRegressionFunction> lbfgsOpt(lrf);
      lbfgsOpt.MaxIterations() = maxIterations;
      lbfgsOpt.MinGradientNorm() = tolerance;
      Log::Info << "Training model with L-BFGS optimizer." << endl;

      // This will train the model.
      LogisticRegression<L_BFGS> lr(lbfgsOpt);
      // Extract the newly trained model.
      model = lr.Parameters();
    }
    else if (optimizerType == "sgd")
    {
      SGD<LogisticRegressionFunction> sgdOpt(lrf);
      sgdOpt.MaxIterations() = maxIterations;
      sgdOpt.Tolerance() = tolerance;
      sgdOpt.StepSize() = stepSize;
      Log::Info << "Training model with SGD optimizer." << endl;

      // This will train the model.
      LogisticRegression<SGD> lr(sgdOpt);
      // Extract the newly trained model.
      model = lr.Parameters();
    }
  }

  if (!testSet.empty())
  {
    // We must perform predictions on the test set.  Training (and the
    // optimizer) are irrelevant here; we'll pass in the model we have.
    LogisticRegression<> lr(model);

    Log::Info << "Predicting classes of points in '" << testFile << "'."
        << endl;
    lr.Predict(testSet, predictions, decisionBoundary);

    // Save the results, if necessary.  Don't transpose.
    if (!outputPredictionsFile.empty())
      data::Save(outputPredictionsFile, predictions, false, false);
  }

  if (!outputFile.empty())
  {
    Log::Info << "Saving model to '" << outputFile << "'." << endl;
    data::Save(outputFile, model, false);
  }
}
