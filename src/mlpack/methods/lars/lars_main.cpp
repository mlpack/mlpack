/**
 * @file lars_main.cpp
 * @author Nishant Mehta
 *
 * Executable for LARS.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "lars.hpp"

PROGRAM_INFO("LARS", "An implementation of LARS: Least Angle Regression "
    "(Stagewise/laSso).  This is a stage-wise homotopy-based algorithm for "
    "L1-regularized linear regression (LASSO) and L1+L2-regularized linear "
    "regression (Elastic Net).\n"
    "\n"
    "This program is able to train a LARS/LASSO/Elastic Net model or load a "
    "model from file, output regression predictions for a test set, and save "
    "the trained model to a file.  The LARS algorithm is described in more "
    "detail below:\n"
    "\n"
    "Let X be a matrix where each row is a point and each column is a "
    "dimension, and let y be a vector of targets.\n"
    "\n"
    "The Elastic Net problem is to solve\n\n"
    "  min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +\n"
    "      0.5 lambda_2 ||beta||_2^2\n\n"
    "If --lambda1 > 0 and --lambda2 = 0, the problem is the LASSO.\n"
    "If --lambda1 > 0 and --lambda2 > 0, the problem is the Elastic Net.\n"
    "If --lambda1 = 0 and --lambda2 > 0, the problem is ridge regression.\n"
    "If --lambda1 = 0 and --lambda2 = 0, the problem is unregularized linear "
    "regression.\n"
    "\n"
    "For efficiency reasons, it is not recommended to use this algorithm with "
    "--lambda_1 = 0.  In that case, use the 'linear_regression' program, which "
    "implements both unregularized linear regression and ridge regression.\n"
    "\n"
    "To train a LARS/LASSO/Elastic Net model, the --input_file and "
    "--responses_file parameters must be given.  The --lambda1 --lambda2, and "
    "--use_cholesky arguments control the training parameters.  A trained model"
    " can be saved with the --output_model_file, or, if training is not desired"
    " at all, a model can be loaded with --input_model_file.  Any output "
    "predictions from a test file can be saved into the file specified by the "
    "--output_predictions option.");

PARAM_TMATRIX_IN("input", "Matrix of covariates (X).", "i");
PARAM_MATRIX_IN("responses", "Matrix of responses/observations (y).", "r");

PARAM_STRING_IN("input_model_file", "File to load model from.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save model to.", "M");

PARAM_TMATRIX_IN("test", "Matrix containing points to regress on (test "
    "points).", "t");

PARAM_TMATRIX_OUT("output_predictions", "If --test_file is specified, this "
    "file is where the predicted responses will be saved.", "o");

PARAM_DOUBLE_IN("lambda1", "Regularization parameter for l1-norm penalty.", "l",
    0);
PARAM_DOUBLE_IN("lambda2", "Regularization parameter for l2-norm penalty.", "L",
    0);
PARAM_FLAG("use_cholesky", "Use Cholesky decomposition during computation "
    "rather than explicitly computing the full Gram matrix.", "c");

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;

int main(int argc, char* argv[])
{
  // Handle parameters,
  CLI::ParseCommandLine(argc, argv);

  double lambda1 = CLI::GetParam<double>("lambda1");
  double lambda2 = CLI::GetParam<double>("lambda2");
  bool useCholesky = CLI::HasParam("use_cholesky");

  // Check parameters -- make sure everything given makes sense.  These checks
  // can be simplified to HasParam() after the reverse compatibility options are
  // removed.
  if (CLI::HasParam("input") && !CLI::HasParam("responses"))
    Log::Fatal << "--input_file (-i) is specified, but --responses_file (-r) is"
        << " not!" << endl;

  if (CLI::HasParam("responses") && !CLI::HasParam("input"))
    Log::Fatal << "--responses_file (-r) is specified, but --input_file (-i) is"
        << " not!" << endl;

  if (!CLI::HasParam("input") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "No input data specified (with --input_file (-i) and "
        << "--responses_file (-r)), and no input model specified (with "
        << "--input_model_file (-m))!" << endl;

  if (CLI::HasParam("input") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Both --input_file (-i) and --input_model_file (-m) are "
        << "specified, but only one may be specified!" << endl;

  if (!CLI::HasParam("output_predictions") &&
      !CLI::HasParam("output_model_file"))
    Log::Warn << "--output_predictions_file (-o) and --output_model_file (-M) "
        << "are not specified; no results will be saved!" << endl;

  if ((CLI::HasParam("output_predictions")) && !CLI::HasParam("test"))
    Log::Warn << "--output_predictions_file (-o) specified, but --test_file "
        << "(-t) is not; no results will be saved." << endl;

  if (CLI::HasParam("test") && !CLI::HasParam("output_predictions"))
    Log::Warn << "--test_file (-t) specified, but --output_predictions_file "
        << "(-o) is not; no results will be saved." << endl;

  // Initialize the object.
  LARS lars(useCholesky, lambda1, lambda2);

  if (CLI::HasParam("input"))
  {
    // Load covariates.  We can avoid LARS transposing our data by choosing to
    // not transpose this data (that's why we used PARAM_TMATRIX_IN).
    mat matX = std::move(CLI::GetParam<arma::mat>("input"));

    // Load responses.  The responses should be a one-dimensional vector, and it
    // seems more likely that these will be stored with one response per line
    // (one per row).  So we should not transpose upon loading.
    mat matY = std::move(CLI::GetParam<arma::mat>("responses"));

    // Make sure y is oriented the right way.
    if (matY.n_rows == 1)
      matY = trans(matY);
    if (matY.n_cols > 1)
      Log::Fatal << "Only one column or row allowed in responses file!" << endl;

    if (matY.n_elem != matX.n_rows)
      Log::Fatal << "Number of responses must be equal to number of rows of X!"
          << endl;

    vec beta;
    lars.Train(matX, matY.unsafe_col(0), beta, false /* do not transpose */);
  }
  else // We must have --input_model_file.
  {
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "lars_model", lars, true);
  }

  if (CLI::HasParam("test"))
  {
    Log::Info << "Regressing on test points." << endl;

    // Load test points.
    mat testPoints = std::move(CLI::GetParam<arma::mat>("test"));

    // Make sure the dimensionality is right.  We haven't transposed, so, we
    // check n_cols not n_rows.
    if (testPoints.n_cols != lars.BetaPath().back().n_elem)
      Log::Fatal << "Dimensionality of test set (" << testPoints.n_cols << ") "
          << "is not equal to the dimensionality of the model ("
          << lars.BetaPath().back().n_elem << ")!" << endl;

    arma::vec predictions;
    lars.Predict(testPoints.t(), predictions, false);

    // Save test predictions.  One per line, so, don't transpose on save.
    if (CLI::HasParam("output_predictions"))
      CLI::GetParam<arma::mat>("output_predictions") = std::move(predictions);
  }

  if (CLI::HasParam("output_model_file"))
  {
    const string outputModelFile = CLI::GetParam<string>("output_model_file");
    data::Save(outputModelFile, "lars_model", lars, true);
  }
}
