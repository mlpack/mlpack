/**
 * @file lars_main.cpp
 * @author Nishant Mehta
 *
 * Executable for LARS.
 */
#include <mlpack/core.hpp>

#include "lars.hpp"

PROGRAM_INFO("LARS", "An implementation of LARS: Least Angle Regression "
    "(Stagewise/laSso).  This is a stage-wise homotopy-based algorithm for "
    "L1-regularized linear regression (LASSO) and L1+L2-regularized linear "
    "regression (Elastic Net).\n"
    "\n"
    "Let X be a matrix where each row is a point and each column is a "
    "dimension, and let y be a vector of targets.\n"
    "\n"
    "The Elastic Net problem is to solve\n\n"
    "  min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +\n"
    "      0.5 lambda_2 ||beta||_2^2\n\n"
    "If lambda_1 > 0 and lambda_2 = 0, the problem is the LASSO.\n"
    "If lambda_1 > 0 and lambda_2 > 0, the problem is the Elastic Net.\n"
    "If lambda_1 = 0 and lambda_2 > 0, the problem is ridge regression.\n"
    "If lambda_1 = 0 and lambda_2 = 0, the problem is unregularized linear "
    "regression.\n"
    "\n"
    "For efficiency reasons, it is not recommended to use this algorithm with "
    "lambda_1 = 0.  In that case, use the 'linear_regression' program, which "
    "implements both unregularized linear regression and ridge regression.\n");

PARAM_STRING_REQ("input_file", "File containing covariates (X).",
    "i");
PARAM_STRING_REQ("responses_file", "File containing y "
    "(responses/observations).", "r");

PARAM_STRING("output_file", "File to save beta (linear estimator) to.", "o",
    "output.csv");

PARAM_STRING("test_file", "File containing points to regress on (test points).",
    "t", "");
PARAM_STRING("output_predictions", "If --test_file is specified, this file is "
    "where the predicted responses will be saved.", "p", "predictions.csv");

PARAM_DOUBLE("lambda1", "Regularization parameter for l1-norm penalty.", "l",
    0);
PARAM_DOUBLE("lambda2", "Regularization parameter for l2-norm penalty.", "L",
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

  // Load covariates.  We can avoid LARS transposing our data by choosing to not
  // transpose this data.
  const string matXFilename = CLI::GetParam<string>("input_file");
  mat matX;
  data::Load(matXFilename, matX, true, false);

  // Load responses.  The responses should be a one-dimensional vector, and it
  // seems more likely that these will be stored with one response per line (one
  // per row).  So we should not transpose upon loading.
  const string yFilename = CLI::GetParam<string>("responses_file");
  mat matY; // Will be a vector.
  data::Load(yFilename, matY, true, false);

  // Make sure y is oriented the right way.
  if (matY.n_rows == 1)
    matY = trans(matY);
  if (matY.n_cols > 1)
    Log::Fatal << "Only one column or row allowed in responses file!" << endl;

  if (matY.n_elem != matX.n_rows)
    Log::Fatal << "Number of responses must be equal to number of rows of X!"
        << endl;

  // Do LARS.
  LARS lars(useCholesky, lambda1, lambda2);
  vec beta;
  lars.Train(matX, matY.unsafe_col(0), beta, false /* do not transpose */);

  const string betaFilename = CLI::GetParam<string>("output_file");
  beta.save(betaFilename, raw_ascii);

  if (CLI::HasParam("test_file"))
  {
    Log::Info << "Regressing on test points." << endl;
    const string testFile = CLI::GetParam<string>("test_file");
    const string outputPredictionsFile =
        CLI::GetParam<string>("output_predictions");

    // Load test points.
    mat testPoints;
    data::Load(testFile, testPoints, true, false);

    arma::vec predictions;
    lars.Predict(testPoints.t(), predictions, false);

    // Save test predictions.  One per line, so, we need a rowvec.
    arma::rowvec predToSave = predictions.t();
    data::Save(outputPredictionsFile, predToSave);
  }
}
